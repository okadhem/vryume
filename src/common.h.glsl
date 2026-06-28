// TODO: Keep in sync with the one in cpp until we use a global define or some solution
const float inter_sample_distance = 0.002; // in m

const float grad_epsilon = 1 * inter_sample_distance;

// given in terms of sdf values, the maximum depth we will reach while computing the gradiant on the surface
const float max_surface_penetration = -sqrt(3) * grad_epsilon; // sqrt(3) is the norm of the k vectors used in the grad

// values are read from the tile by interpolating 8 surrounding sdf values, all these values
// should be representable. therefore we need to represent the sdf of the point along the
// diagonal of the max_surface_penetration, that is the "-sqrt(3) * inter_sample_distance" term.
const float min_representable_distance = max_surface_penetration + (-sqrt(3) * inter_sample_distance);
const float encoded_distance_resolution = 0.0002; // in m
const float max_representable_distance = min_representable_distance + 255.0 * encoded_distance_resolution;

// inverse of glsl mix function
// given x in [a,b] return the interpolation parameter in range [0, 1]
// it must be the case that: a < b and x in [a,b]
float inverse_mix(float a, float b, float x)
{
    return (x - a) / (b - a);
}

float encode_distance(float distance) {
    float clamped = clamp(distance, min_representable_distance, max_representable_distance);
    return inverse_mix(min_representable_distance, max_representable_distance, clamped);
}

float decode_distance(float sampled_distance) {
    return mix(min_representable_distance, max_representable_distance,
        sampled_distance);
}

uint norm_float_to_4bit_uint(float v) {
    return uint(round(v * 15.0));
}

const float max_representable_distance_4bits = sqrt(3) * inter_sample_distance;
const float min_representable_distance_4bits = -max_representable_distance_4bits;

float encode_distance_4bits(float distance) {
    // not sure clamping is needed here, algorithm only encodes distances near the surfaces.
    float clamped = clamp(distance, min_representable_distance_4bits, max_representable_distance_4bits);
    return inverse_mix(min_representable_distance_4bits, max_representable_distance_4bits, clamped);
}
float decode_distance_4bits(float distance) {
    return mix(min_representable_distance_4bits, max_representable_distance_4bits, distance);
}

const float MIN_FLOAT = -3.402823e38;

const uint UINT_MAX = 0xFFFFFFFFu;
