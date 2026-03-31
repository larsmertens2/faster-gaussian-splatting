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

        return make_float3(1.0f,1.0f,0.0f);

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
    }
       

}
