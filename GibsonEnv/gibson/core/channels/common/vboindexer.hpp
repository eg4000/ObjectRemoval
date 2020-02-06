#ifndef VBOINDEXER_HPP
#define VBOINDEXER_HPP

void indexVBO(
    std::vector<glm::vec3> & in_vertices,
    std::vector<glm::vec2> & in_uvs,
    std::vector<glm::vec3> & in_normals,

    std::vector<unsigned int> & out_indices,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals
);

void indexVBO_MTL(
    std::vector<std::vector<glm::vec3>> & mtl_vertices,
    std::vector<std::vector<glm::vec2>> & mtl_uvs,
    std::vector<std::vector<glm::vec3>> & mtl_normals,

    std::vector<unsigned int> & out_indices,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals,
    std::vector<glm::vec2> & out_semantics
);

void indexVBO_PLY(
    std::vector<std::vector<glm::vec3>> & mtl_vertices,
    std::vector<std::vector<glm::vec2>> & mtl_uvs,
    std::vector<std::vector<glm::vec3>> & mtl_normals,

    std::vector<unsigned int> & out_indices,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals,
    std::vector<glm::vec2> & out_semantics
);

void indexVBO_TBN(
    std::vector<glm::vec3> & in_vertices,
    std::vector<glm::vec2> & in_uvs,
    std::vector<glm::vec3> & in_normals,
    std::vector<glm::vec3> & in_tangents,
    std::vector<glm::vec3> & in_bitangents,

    std::vector<unsigned int> & out_indices,
    std::vector<glm::vec3> & out_vertices,
    std::vector<glm::vec2> & out_uvs,
    std::vector<glm::vec3> & out_normals,
    std::vector<glm::vec3> & out_tangents,
    std::vector<glm::vec3> & out_bitangents
);

#endif