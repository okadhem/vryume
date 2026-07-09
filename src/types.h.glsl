struct Edit {
    bool is_removal;
    int material_id;
    uint primitive_type;
    float param[4];

    // transform in model space
    vec4 rotation; // unit quaternion
    vec3 translation;
};

const uint EDIT_PRIMITIVE_SPHERE = 1;
const uint EDIT_PRIMITIVE_BOX = 2;
