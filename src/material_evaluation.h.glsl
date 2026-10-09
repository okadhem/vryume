// Each voxel material byte is made of a header and a payload:
// header (2 bits): indicate the mapping of the material_id in payload, either explicitly, or mark it as "default" mapping
// which is mapping of materials in voxels away from 3 surface voxels and their neighbours
// payload (6 bits): material_id

const uint HEADER_MASK = 0xC0u; // 0b11000000u;

const uint DEFAULT_MAPPING_HEADER = 0u;
const uint CHANNEL_1_MAPPING_HEADER = 0x40u; //0b01000000
const uint CHANNEL_2_MAPPING_HEADER = 0x80u; //0b10000000
const uint CHANNEL_3_MAPPING_HEADER = 0x60u; //0b11000000

uint make_material_info(uint material_id, uint channel_index) {
    switch (channel_index) {
        case 0:
        return CHANNEL_1_MAPPING_HEADER & material_id;
        break;
        case 1:
        return CHANNEL_2_MAPPING_HEADER & material_id;
        break;
        case 2:
        return CHANNEL_3_MAPPING_HEADER & material_id;
        break;
        default:
        //panic
        return UINT_MAX;
    }
}

uint material_id_from_info(uint material_info) {
    return material_info & (~HEADER_MASK);
}

// converts between coordinates centred around a voxel, ranging from -1 to 1 and linearized id.
uint coord_to_neighbour_id(ivec3 c) {
    // coordinates relative to the lower corner "neighbour" in the 3x3 block.
    uint x_corner = c.x + 1;
    uint y_corner = c.y + 1;
    uint z_corner = c.z + 1;
    return x_corner + 3 * y_corner + 9 * z_corner;
}
// converts between coordinates centred around a voxel, ranging from -1 to 1 and linearized id.
ivec3 neighbour_id_to_coord(uint neighbour_id) {
    uint x_corner = neighbour_id % 3;
    uint y_corner = (neighbour_id / 3) % 3;
    uint z_corner = neighbour_id / 9;
    return ivec3(x_corner, y_corner, z_corner) + ivec3(-1, -1, -1);
}
