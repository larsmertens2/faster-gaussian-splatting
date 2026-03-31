#pragma once

#include "helper_math.h"

namespace faster_gs::rasterization::kernels {

    __device__ inline float3 convert_sg_to_color(
        const float3& Amplitude,    // α (RGB)
        const float3& Axis,         // µ (Axis for lobe)
        const float& Sharpness,    // λ (Sharpness)
        const float3& position,     // d (Position)
        const float3& cam_position, // d(Position)
        const uint primitive_idx    // id Gaussian
    )
    {
        
        float3 d = normalize(position - cam_position);

        float cosAngle = dot(d, Axis);

        float factor = expf(Sharpness * (cosAngle - 1.0f)); //G(v; µ, λ, α) = α * exp(λ * (dot(v, µ) - 1))

        float3 color = Amplitude * factor;

        return make_float3(0.0f,1.0f,0.0f);

    }


    //C = αe^λ(d·µ−1)
    // dC/dα = e^λ(d·µ−1)
    // dC/dλ = α_(d·µ−1)_*e^λ(d·µ−1)
    // dC/dµ = αλd e^λ(d·µ-1)
    //dC/dd = αλµ e^λ(d·µ-1)

    __device__ inline float3 convert_sg_to_color_backward(
        const float3& Amplitude,    // α (RGB)
        const float3& Axis,         // µ (Axis for lobe)
        const float& Sharpness,     // λ (Sharpness)
        const float3& position,     // Positie van de Gaussian
        const float3& cam_position, // Positie van de camera
        const uint primitive_idx,   // Index (id Gaussian)
        const float3& grad_color,   // dLoss/dC (Input van de loss)

        float3* grad_Amplitude,     // dL/dα (Output pointer)
        float3* grad_Lobe,          // dL/dµ (Output pointer)
        float* grad_Sharpness       // dL/dλ (Output pointer)
    )
    {

        auto [x_raw, y_raw, z_raw] = position - cam_position; //diff camera and Gaussian

        //distances squared
        const float xx_raw = x_raw * x_raw;
        const float yy_raw = y_raw * y_raw;
        const float zz_raw = z_raw * z_raw;
        const float xy_raw = x_raw * y_raw;
        const float xz_raw = x_raw * z_raw;
        const float yz_raw = y_raw * z_raw;

        const float norm_sq = xx_raw + yy_raw + zz_raw; // L^2
        const float inv_dist = rsqrtf(norm_sq);         // 1/L
        const float3 dir = make_float3(x_raw * inv_dist, y_raw * inv_dist, z_raw * inv_dist); // d

        // Forward pass termen herhalen
        float cosAngle = dot(dir, Axis);
        float exponential_function = expf(Sharpness * (cosAngle - 1.0f));

        *grad_Amplitude = exponential_function * grad_color; // α gradient

        // We gebruiken het dot product (dL/dC · α) als scalar basis
        float dL_dC_dot_alpha = dot(grad_color, Amplitude); 
        float common_factor = dL_dC_dot_alpha * exponential_function;

        *grad_Sharpness = common_factor * (cosAngle - 1.0f); // λ gradient

        // Raw gradient voor de as (µ), inclusief projectie om unit-vector te blijven
        float3 raw_grad_mu = (common_factor * Sharpness) * dir;
        *grad_Lobe = raw_grad_mu - dot(raw_grad_mu, Axis) * Axis; // µ gradient

        // 5. dLoss/dPosition 
        // Eerst de gradient ten opzichte van de richting 'd'
        float3 grad_direction = (common_factor * Sharpness) * Axis; // dL/dd = dL/dC * dC/dd

        // De Jacobian matrix vermenigvuldiging: g_pos = (1/L^3) * [Matrix] * g_dir
        const float inv_L3 = rsqrtf(norm_sq * norm_sq * norm_sq);
        
        float3 dcolor_dposition; //verplaatsing camera voor betere kleur
        dcolor_dposition.x = (yy_raw + zz_raw) * grad_direction.x 
                           - xy_raw * grad_direction.y 
                           - xz_raw * grad_direction.z;

        dcolor_dposition.y = -xy_raw * grad_direction.x 
                           + (xx_raw + zz_raw) * grad_direction.y 
                           - yz_raw * grad_direction.z;

        dcolor_dposition.z = -xz_raw * grad_direction.x 
                           - yz_raw * grad_direction.y 
                           + (xx_raw + yy_raw) * grad_direction.z;

        return dcolor_dposition * inv_L3;
    }
       

}
