#include <metal_stdlib>

using namespace metal;

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define CHANNELS 3
#define MAX_REGISTER_CHANNELS 3

constant float SH_C0 = 0.28209479177387814f;
constant float SH_C1 = 0.4886025119029199f;
constant float SH_C2[] = {
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f};
constant float SH_C3[] = {
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f};
constant float SH_C4[] = {
    2.5033429417967046f,
    -1.7701307697799304,
    0.9461746957575601f,
    -0.6690465435572892f,
    0.10578554691520431f,
    -0.6690465435572892f,
    0.47308734787878004f,
    -1.7701307697799304f,
    0.6258357354491761f};

inline uint num_sh_bases(const uint degree) {
    if (degree == 0)
        return 1;
    if (degree == 1)
        return 4;
    if (degree == 2)
        return 9;
    if (degree == 3)
        return 16;
    return 25;
}

inline float ndc2pix(const float x, const float W, const float cx) {
    return 0.5f * W * x + cx - 0.5f;
}

inline void get_bbox(
    const float2 center,
    const float2 dims,
    const int3 img_size,
    thread uint2 &bb_min,
    thread uint2 &bb_max
) {
    // get bounding box with center and dims, within bounds
    // bounding box coords returned in tile coords, inclusive min, exclusive max
    // clamp between 0 and tile bounds
    bb_min.x = min(max(0, (int)(center.x - dims.x)), img_size.x);
    bb_max.x = min(max(0, (int)(center.x + dims.x + 1)), img_size.x);
    bb_min.y = min(max(0, (int)(center.y - dims.y)), img_size.y);
    bb_max.y = min(max(0, (int)(center.y + dims.y + 1)), img_size.y);
}

inline void get_tile_bbox(
    const float2 pix_center,
    const float pix_radius,
    const int3 tile_bounds,
    thread uint2 &tile_min,
    thread uint2 &tile_max
) {
    // gets gaussian dimensions in tile space, i.e. the span of a gaussian in
    // tile_grid (image divided into tiles)
    float2 tile_center = {
        pix_center.x / (float)BLOCK_X, pix_center.y / (float)BLOCK_Y
    };
    float2 tile_radius = {
        pix_radius / (float)BLOCK_X, pix_radius / (float)BLOCK_Y
    };
    get_bbox(tile_center, tile_radius, tile_bounds, tile_min, tile_max);
}

// helper for applying R * p + T, expect mat to be ROW MAJOR
inline float3 transform_4x3(constant float *mat, const float3 p) {
    float3 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
    };
    return out;
}

// helper to apply 4x4 transform to 3d vector, return homo coords
// expects mat to be ROW MAJOR
inline float4 transform_4x4(constant float *mat, const float3 p) {
    float4 out = {
        mat[0] * p.x + mat[1] * p.y + mat[2] * p.z + mat[3],
        mat[4] * p.x + mat[5] * p.y + mat[6] * p.z + mat[7],
        mat[8] * p.x + mat[9] * p.y + mat[10] * p.z + mat[11],
        mat[12] * p.x + mat[13] * p.y + mat[14] * p.z + mat[15],
    };
    return out;
}

inline float3x3 quat_to_rotmat(const float4 quat) {
    // Corrected quaternion normalization and component assignment
    // to prevent numerical issues.
    float norm_sq = quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w;
    float s = (norm_sq > 1e-8f) ? (2.0f / norm_sq) : 0.0f;

    float x = quat.x;
    float y = quat.y;
    float z = quat.z;
    float w = quat.w;

    float xx = x * x * s;
    float yy = y * y * s;
    float zz = z * z * s;
    float xy = x * y * s;
    float xz = x * z * s;
    float yz = y * z * s;
    float wx = w * x * s;
    float wy = w * y * s;
    float wz = w * z * s;

    // Metal matrices are column-major
    return float3x3(
        1.f - (yy + zz),
        xy - wz,
        xz + wy,
        xy + wz,
        1.f - (xx + zz),
        yz - wx,
        xz - wy,
        yz + wx,
        1.f - (xx + yy)
    );
}

// device helper for culling near points
inline bool clip_near_plane(
    const float3 p,
    constant float *viewmat,
    thread float3 &p_view,
    float thresh
) {
    p_view = transform_4x3(viewmat, p);
    if (p_view.z <= thresh) {
        return true;
    }
    return false;
}

inline float3x3 scale_to_mat(const float3 scale, const float glob_scale) {
    float3x3 S = float3x3(1.f);
    S[0][0] = glob_scale * scale.x;
    S[1][1] = glob_scale * scale.y;
    S[2][2] = glob_scale * scale.z;
    return S;
}

// device helper to get 3D covariance from scale and quat parameters
inline void scale_rot_to_cov3d(
    const float3 scale, const float glob_scale, const float4 quat, device float *cov3d
) {
    float3x3 R = quat_to_rotmat(quat);
    float3x3 S = scale_to_mat(scale, glob_scale);

    float3x3 M = R * S;
    float3x3 tmp = M * transpose(M);

    // Save upper triangular part because the matrix is symmetric
    cov3d[0] = tmp[0][0];
    cov3d[1] = tmp[0][1];
    cov3d[2] = tmp[0][2];
    cov3d[3] = tmp[1][1];
    cov3d[4] = tmp[1][2];
    cov3d[5] = tmp[2][2];
}

// device helper to approximate projected 2d cov from 3d mean and cov
float3 project_cov3d_ewa(
    thread float3& mean3d,
    device float* cov3d,
    constant float* viewmat,
    const float fx,
    const float fy,
    const float tan_fovx,
    const float tan_fovy
) {
    // We expect row-major matrices as input, Metal uses column-major
    // Upper 3x3 submatrix
    float3x3 W = float3x3(
        viewmat[0],
        viewmat[4],
        viewmat[8],
        viewmat[1],
        viewmat[5],
        viewmat[9],
        viewmat[2],
        viewmat[6],
        viewmat[10]
    );
    float3 p = float3(viewmat[3], viewmat[7], viewmat[11]);
    float3 t = W * float3(mean3d.x, mean3d.y, mean3d.z) + p;

    // Clip so that the covariance stays within reasonable bounds
    float lim_x = 1.3f * tan_fovx;
    float lim_y = 1.3f * tan_fovy;
    float t_z_safe = max(1e-6f, t.z); // Avoid division by zero
    t.x = t_z_safe * min(lim_x, max(-lim_x, t.x / t_z_safe));
    t.y = t_z_safe * min(lim_y, max(-lim_y, t.y / t_z_safe));

    float rz = 1.f / t_z_safe;
    float rz2 = rz * rz;

    // Column-major
    // We only care about the top 2x2 submatrix
    float3x3 J = float3x3(
        fx * rz,
        0.f,
        0.f,
        0.f,
        fy * rz,
        0.f,
        -fx * t.x * rz2,
        -fy * t.y * rz2,
        0.f
    );
    float3x3 T = J * W;

    float3x3 V = float3x3(
        cov3d[0],
        cov3d[1],
        cov3d[2],
        cov3d[1],
        cov3d[3],
        cov3d[4],
        cov3d[2],
        cov3d[4],
        cov3d[5]
    );

    float3x3 cov = T * V * transpose(T);

    // Add a small blur along axes and save upper triangular elements
    return float3(float(cov[0][0]) + 0.3f, float(cov[0][1]), float(cov[1][1]) + 0.3f);
}

inline bool compute_cov2d_bounds(
    const float3 cov2d,
    thread float3 &conic,
    thread float &radius
) {
    // Find eigenvalues of 2D covariance matrix
    // Expects upper triangular values of cov matrix as float3
    // Then compute the radius and conic dimensions
    // The conic is the inverse cov2d matrix, represented here with upper
    // triangular values.
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (fabs(det) < 1e-8f)
        return false;
    float inv_det = 1.f / det;

    // Inverse of 2x2 cov2d matrix
    conic.x = cov2d.z * inv_det;
    conic.y = -cov2d.y * inv_det;
    conic.z = cov2d.x * inv_det;

    float b = 0.5f * (cov2d.x + cov2d.z);
    float discrim = b * b - det;
    float sqrt_discrim = sqrt(max(0.0f, discrim));
    float v1 = b + sqrt_discrim;
    float v2 = b - sqrt_discrim;

    // Take 3 sigma of covariance
    float max_eigenvalue = max(fabs(v1), fabs(v2));
    radius = ceil(3.f * sqrt(max(1e-8f, max_eigenvalue)));
    return true;
}

inline float2 project_pix(
    constant float *mat, const float3 p, const uint2 img_size, const float2 pp
) {
    // ROW MAJOR mat
    float4 p_hom = transform_4x4(mat, p);
    float rw = 1.f / (p_hom.w + 1e-6f); // Avoid division by zero
    float3 p_proj = {p_hom.x * rw, p_hom.y * rw, p_hom.z * rw};
    return {
        ndc2pix(p_proj.x, (int)img_size.x, pp.x), ndc2pix(p_proj.y, (int)img_size.y, pp.y)
    };
}

/* 
    !!!!IMPORTANT!!!
    Metal does not support packed arrays of vectorized types like int2, float2, float3, etc.
    and instead pads the elements of arrays of these types to fixed alignments. 
    Use the below functions to read and write from packed arrays of these types.
*/

inline int2 read_packed_int2(constant int* arr, int idx) {
    return int2(arr[2*idx], arr[2*idx+1]);
}

inline void write_packed_int2(device int* arr, int idx, int2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

inline void write_packed_int2x(device int* arr, int idx, int x) {
    arr[2*idx] = x;
}

inline void write_packed_int2y(device int* arr, int idx, int y) {
    arr[2*idx+1] = y;
}

inline float2 read_packed_float2(constant float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

inline float2 read_packed_float2(device float* arr, int idx) {
    return float2(arr[2*idx], arr[2*idx+1]);
}

inline void write_packed_float2(device float* arr, int idx, float2 val) {
    arr[2*idx] = val.x;
    arr[2*idx+1] = val.y;
}

inline int3 read_packed_int3(constant int* arr, int idx) {
    return int3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline void write_packed_int3(device int* arr, int idx, int3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

inline float3 read_packed_float3(constant float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline float3 read_packed_float3(device float* arr, int idx) {
    return float3(arr[3*idx], arr[3*idx+1], arr[3*idx+2]);
}

inline void write_packed_float3(device float* arr, int idx, float3 val) {
    arr[3*idx] = val.x;
    arr[3*idx+1] = val.y;
    arr[3*idx+2] = val.z;
}

inline float4 read_packed_float4(constant float* arr, int idx) {
    return float4(arr[4*idx], arr[4*idx+1], arr[4*idx+2], arr[4*idx+3]);
}

inline void write_packed_float4(device float* arr, int idx, float4 val) {
    arr[4*idx] = val.x;
    arr[4*idx+1] = val.y;
    arr[4*idx+2] = val.z;
    arr[4*idx+3] = val.w;
}

// Kernel functions and other code remain mostly unchanged, except for adjustments
// to prevent numerical issues as identified.

...

// In the sh_coeffs_to_color function, prevent division by zero
void sh_coeffs_to_color(
    const uint degree,
    const float3 viewdir,
    constant float *coeffs,
    device float *colors
) {
    // Expects v_colors to be len CHANNELS
    // and v_coeffs to be num_bases * CHANNELS
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] = SH_C0 * coeffs[c];
    }
    if (degree < 1) {
        return;
    }

    float norm_sq = viewdir.x * viewdir.x + viewdir.y * viewdir.y + viewdir.z * viewdir.z;
    float norm = sqrt(max(1e-8f, norm_sq)); // Prevent division by zero
    float x = viewdir.x / norm;
    float y = viewdir.y / norm;
    float z = viewdir.z / norm;

    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    // Supports up to num_bases = 25
    for (int c = 0; c < CHANNELS; ++c) {
        colors[c] += SH_C1 * (-y * coeffs[1 * CHANNELS + c] +
                              z * coeffs[2 * CHANNELS + c] -
                              x * coeffs[3 * CHANNELS + c]);
        if (degree < 2) {
            continue;
        }
        colors[c] +=
            (SH_C2[0] * xy * coeffs[4 * CHANNELS + c] +
             SH_C2[1] * yz * coeffs[5 * CHANNELS + c] +
             SH_C2[2] * (2.f * zz - xx - yy) * coeffs[6 * CHANNELS + c] +
             SH_C2[3] * xz * coeffs[7 * CHANNELS + c] +
             SH_C2[4] * (xx - yy) * coeffs[8 * CHANNELS + c]);
        if (degree < 3) {
            continue;
        }
        // Continue with higher-degree terms as in the original code...
    }
}

...

// Similar changes should be made in other functions where division by zero or
// square roots of negative numbers could occur.

