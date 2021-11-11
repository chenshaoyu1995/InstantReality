// Preprocessor Directives
#define STB_IMAGE_IMPLEMENTATION
#define ASSET_PATH "/Glitter/Assets/"

// Local Headers
#include "mesh.hpp"
#include "utility.hpp"

// System Headers
#include <stb_image.h>
#include <algorithm>
#include <glm/gtx/transform.hpp> 
#include <glm/gtc/type_ptr.hpp>

// Define Namespace
namespace Mirage
{

    Mesh::Mesh(std::vector<Vertex> vertices,
        std::vector<GLuint> indices,
        std::map<GLuint, std::string> textures)
        : mIndices(std::move(indices))
        , mVertices(std::move(vertices))
        , mTextures(std::move(textures))
    {}

    bool Mesh::isTexture16(std::string filename)
    {
        filename = PROJECT_SOURCE_DIR ASSET_PATH + filename;
        return stbi_is_16_bit(filename.c_str());
    }

    GLuint Mesh::loadTexture(std::string filename)
    {
        GLenum format;
        GLuint texture;

        int width, height, channels;
        filename = PROJECT_SOURCE_DIR ASSET_PATH + filename;
        unsigned char* image = stbi_load(filename.c_str(), &width, &height, &channels, 0);
        if (!image) fprintf(stderr, "%s %s\n", "Failed to Load Texture", filename.c_str());
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        // Set the Correct Channel Format
        switch (channels)
        {
        case 1: format = GL_RED;       break;
        case 2: format = GL_RG;        break;
        case 3: format = GL_RGB;       break;
        case 4: format = GL_RGBA;      break;
        }

        // Bind Texture and Set Filtering Levels
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, format,
            width, height, 0, format, GL_UNSIGNED_BYTE, image);
        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(image);

        return texture;
    }

    GLuint Mesh::loadTexture16(std::string filename)
    {
        GLenum sizedFormat;
        GLenum format;
        GLuint texture;

        int width, height, channels;
        filename = PROJECT_SOURCE_DIR ASSET_PATH + filename;
        unsigned short* image = stbi_load_16(filename.c_str(), &width, &height, &channels, 0);
        if (!image) fprintf(stderr, "%s %s\n", "Failed to Load Texture", filename.c_str());

        // Set the Correct Channel Format
        switch (channels)
        {
        case 1: format = GL_RED; sizedFormat = GL_R16;      break;
        case 2: format = GL_RG; sizedFormat = GL_RG16;      break;
        case 3: format = GL_RGB; sizedFormat = GL_RGB16;    break;
        case 4: format = GL_RGBA; sizedFormat = GL_RGBA16;  break;
        }

        // Bind Texture and Set Filtering Levels
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, sizedFormat,
            width, height, 0, format, GL_UNSIGNED_SHORT, image);
        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(image);

        return texture;
    }

    GLuint Mesh::loadAiTexture(const aiTexture* tex)
    {
        GLenum format;
        GLuint texture;

        int width, height, channels;
        unsigned char* image = stbi_load_from_memory((unsigned char*)tex->pcData, tex->mWidth, &width, &height, &channels, 0);
        if (!image) fprintf(stderr, "%s\n", "Failed to Load Texture");

        // Set the Correct Channel Format
        switch (channels)
        {
        case 1: format = GL_ALPHA;     break;
        case 2: format = GL_LUMINANCE; break;
        case 3: format = GL_RGB;       break;
        case 4: format = GL_RGBA;      break;
        }

        // Bind Texture and Set Filtering Levels
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, format,
            width, height, 0, format, GL_UNSIGNED_BYTE, image);
        glGenerateMipmap(GL_TEXTURE_2D);

        stbi_image_free(image);

        return texture;
    }

    std::map<GLuint, std::string> Mesh::process(std::string const& path,
        aiMaterial* material,
        aiTextureType type, aiScene const* scene)
    {
        std::string mode;
        std::map<GLuint, std::string> textures;
        GLuint texture;

        for (unsigned int i = 0; i < material->GetTextureCount(type); i++)
        {
            aiString str; material->GetTexture(type, i, &str);
            if (scene->GetEmbeddedTexture(str.C_Str()) == nullptr) {
                if (isTexture16(path + str.C_Str())) texture = loadTexture16(path + str.C_Str());
                else texture = loadTexture(path + str.C_Str());
            }
            else {
                const aiTexture* tex = scene->GetEmbeddedTexture(str.C_Str());
                if (tex->mHeight != 0) {
                    GLuint texture;

                    // Bind Texture and Set Filtering Levels
                    glGenTextures(1, &texture);
                    glBindTexture(GL_TEXTURE_2D, texture);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                        tex->mWidth, tex->mHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex->pcData);
                    glGenerateMipmap(GL_TEXTURE_2D);
                }
                else {
                    texture = loadAiTexture(tex);
                }
            }

            if (type == aiTextureType_DIFFUSE)  mode = "diffuse";
            else if (type == aiTextureType_SPECULAR) mode = "specular";
            else if (type == aiTextureType_EMISSIVE) mode = "emissive";
            else if (type == aiTextureType_NORMALS) mode = "normal";
            else if (type == aiTextureType_DISPLACEMENT) mode = "displacement";
            textures.insert(std::make_pair(texture, mode));
        }   return textures;
    }

    void Mesh::parseNode(std::string const & path, aiNode const * node, aiScene const * scene)
    {
        for (unsigned int i = 0; i < node->mNumMeshes; i++)
            parse(path, scene->mMeshes[node->mMeshes[i]], scene);
        for (unsigned int i = 0; i < node->mNumChildren; i++)
            parseNode(path, node->mChildren[i], scene);
    }

    TexMesh::TexMesh(std::string const& filename)
    {
        // Load a Model from File
        Assimp::Importer loader;
        aiScene const* scene = loader.ReadFile(
            PROJECT_SOURCE_DIR ASSET_PATH + filename,
            aiProcess_Triangulate | aiProcess_FlipUVs);

        // Walk the Tree of Scene Nodes
        auto index = filename.find_last_of("/");
        if (!scene) fprintf(stderr, "%s\n", loader.GetErrorString());
        else parseNode((index != std::string::npos) ? filename.substr(0, index + 1) : "", scene->mRootNode, scene);
    }

    TexMesh::TexMesh(std::vector<Vertex> vertices, std::vector<GLuint> indices, std::map<GLuint, std::string> textures) :
        Mesh(vertices, indices, textures)
    {
        // Bind a Vertex Array Object
        glGenVertexArrays(1, &mVertexArray);
        glBindVertexArray(mVertexArray);

        // Copy Vertex Buffer Data
        glGenBuffers(1, &mVertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
        glBufferData(GL_ARRAY_BUFFER,
            mVertices.size() * sizeof(Vertex),
            &mVertices.front(), GL_STATIC_DRAW);

        // Copy Index Buffer Data
        glGenBuffers(1, & mElementBuffer);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mElementBuffer);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     mIndices.size() * sizeof(GLuint),
                   & mIndices.front(), GL_STATIC_DRAW);

        // Set Shader Attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, uv));
        glEnableVertexAttribArray(0); // Vertex Positions
        glEnableVertexAttribArray(1); // Vertex Normals
        glEnableVertexAttribArray(2); // Vertex UVs

        // Cleanup Buffers
        glBindVertexArray(0);
    }

    void TexMesh::draw(GLuint shader)
    {
        unsigned int unit = 0;
        for (auto& i : mSubMeshes) i->draw(shader);
        for (auto& i : mTextures)
        {   // Set Correct Uniform Names Using Texture Type (Omit ID for 0th Texture)
            std::string uniform = i.second;

            // Bind Correct Textures and Vertex Array Before Drawing
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, i.first);
            glUniform1i(glGetUniformLocation(shader, uniform.c_str()), unit++);
        }   glBindVertexArray(mVertexArray);
        glUniform1i(glGetUniformLocation(shader, "hasNoTexture"), mNoTexture);
        if (mNoTexture) glUniform3fv(glGetUniformLocation(shader, "diffuseColor"), 1, glm::value_ptr(mDiffuseColor));

        glDrawElements(GL_TRIANGLES, mIndices.size(), GL_UNSIGNED_INT, 0);

        for (unsigned int i = 0; i < unit; ++i) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }

    void TexMesh::parse(std::string const& path, aiMesh const* mesh, aiScene const* scene)
    {
        // Create Vertex Data from Mesh Node
        std::vector<Vertex> vertices; Vertex vertex;
        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            if (mesh->mTextureCoords[0]) vertex.uv = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
            vertex.position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
            vertex.normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
            vertices.push_back(vertex);
        }

        // Create Mesh Indices for Indexed Drawing
        std::vector<GLuint> indices;
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
            for (unsigned int j = 0; j < mesh->mFaces[i].mNumIndices; j++)
                indices.push_back(mesh->mFaces[i].mIndices[j]);

        // Load Mesh Textures into VRAM
        std::map<GLuint, std::string> textures;
        auto diffuse = process(path, scene->mMaterials[mesh->mMaterialIndex], aiTextureType_DIFFUSE, scene);
        textures.insert(diffuse.begin(), diffuse.end());

        auto i = scene->mMaterials[mesh->mMaterialIndex];
        aiColor3D color;
        i->Get(AI_MATKEY_COLOR_DIFFUSE, color);

        // Create New Mesh Node
        mSubMeshes.push_back(std::shared_ptr<Mesh>(new TexMesh(std::move(vertices), std::move(indices), std::move(textures))));
        auto p = std::dynamic_pointer_cast<TexMesh>(mSubMeshes.back());
        if (p->mTextures.empty()) {
            p->mNoTexture = true;
            p->mDiffuseColor = { color.r, color.g, color.b };
        }
    }

    TessellationMesh::TessellationMesh(std::string const& filename)
    {
        // Load a Model from File
        Assimp::Importer loader;
        aiScene const* scene = loader.ReadFile(
            PROJECT_SOURCE_DIR ASSET_PATH + filename,
            aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices | aiProcess_CalcTangentSpace);

        // Walk the Tree of Scene Nodes
        auto index = filename.find_last_of("/");
        if (!scene) fprintf(stderr, "%s\n", loader.GetErrorString());
        else parseNode((index != std::string::npos) ? filename.substr(0, index + 1) : "", scene->mRootNode, scene);
    }

    TessellationMesh::TessellationMesh(std::vector<Vertex> vertices, std::vector<GLuint> indices, std::map<GLuint, std::string> textures, std::vector<glm::vec3> tangents, size_t vertSize):
        Mesh(std::move(vertices), std::move(indices), std::move(textures)),
        mTangents(std::move(tangents)),
        mVertSize(vertSize)
    {
        // Bind a Vertex Array Object
        glGenVertexArrays(1, &mVertexArray);
        glBindVertexArray(mVertexArray);

        // Copy Vertex Buffer Data
        glGenBuffers(1, &mVertexBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
        glBufferData(GL_ARRAY_BUFFER,
            mVertices.size() * sizeof(Vertex),
            &mVertices.front(), GL_STATIC_DRAW);

        // Set Shader Attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (GLvoid*)offsetof(Vertex, uv));
        glEnableVertexAttribArray(0); // Vertex Positions
        glEnableVertexAttribArray(1); // Vertex Normals
        glEnableVertexAttribArray(2); // Vertex UVs

        glGenBuffers(1, &mTangentBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, mTangentBuffer);
        glBufferData(GL_ARRAY_BUFFER,
            mTangents.size() * sizeof(glm::vec3),
            &mTangents.front(), GL_STATIC_DRAW);

        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(3); // Vertex Tangents

        // Cleanup Buffers
        glBindVertexArray(0);
    }

    void TessellationMesh::parse(std::string const& path, aiMesh const* mesh, aiScene const* scene)
    {
        // Create Vertex Data from Mesh Node
        std::vector<Vertex> vertices; Vertex vertex;
        for (unsigned int i = 0; i < mesh->mNumVertices; i++)
        {
            if (mesh->mTextureCoords[0]) vertex.uv = glm::vec2(mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y);
            vertex.position = glm::vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z);
            vertex.normal = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);
            vertices.push_back(vertex);
        }

        // Create Mesh Indices for Indexed Drawing
        std::vector<GLuint> indices;
        for (unsigned int i = 0; i < mesh->mNumFaces; i++)
            for (unsigned int j = 0; j < mesh->mFaces[i].mNumIndices; j++)
                indices.push_back(mesh->mFaces[i].mIndices[j]);

        std::vector<Vertex> flatVertices;
        for (auto i : indices) flatVertices.push_back(vertices[i]);

        std::vector<glm::vec3> tangents;
        for (auto i : indices) tangents.emplace_back(mesh->mTangents[i].x, mesh->mTangents[i].y, mesh->mTangents[i].z);

        // Load Mesh Textures into VRAM
        std::map<GLuint, std::string> textures;
        auto diffuse = process(path, scene->mMaterials[mesh->mMaterialIndex], aiTextureType_DIFFUSE, scene);
        auto specular = process(path, scene->mMaterials[mesh->mMaterialIndex], aiTextureType_SPECULAR, scene);
        auto emissive = process(path, scene->mMaterials[mesh->mMaterialIndex], aiTextureType_EMISSIVE, scene);
        auto normal = process(path, scene->mMaterials[mesh->mMaterialIndex], aiTextureType_NORMALS, scene);
        auto displacement = process(path, scene->mMaterials[mesh->mMaterialIndex], aiTextureType_DISPLACEMENT, scene);
        textures.insert(diffuse.begin(), diffuse.end());
        textures.insert(specular.begin(), specular.end());
        textures.insert(emissive.begin(), emissive.end());
        textures.insert(normal.begin(), normal.end());
        textures.insert(displacement.begin(), displacement.end());

        // Create New Mesh Node
        mSubMeshes.push_back(std::shared_ptr<Mesh>(new TessellationMesh(std::move(flatVertices), std::move(indices), std::move(textures), std::move(tangents), vertices.size())));
        if (displacement.empty()) std::cout << "Not displacment map found for: " << path << std::endl;
    }

    void TessellationMesh::SetupTesBuffer(GLuint faceTesBuffer, GLuint edgeTesBuffer) {
        mFaceTesBuffer = faceTesBuffer;
        mEdgeTesBuffer = edgeTesBuffer;
        glBindVertexArray(mVertexArray);
        glBindBuffer(GL_ARRAY_BUFFER, mFaceTesBuffer);
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(4);
        glBindBuffer(GL_ARRAY_BUFFER, mEdgeTesBuffer);
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glEnableVertexAttribArray(5);
        glBindVertexArray(0);
    }

    void TessellationMesh::draw(GLuint shader) {
        unsigned int unit = 0;
        for (auto& i : mSubMeshes) i->draw(shader);
        for (auto& i : mTextures)
        {   // Set Correct Uniform Names Using Texture Type (Omit ID for 0th Texture)
            std::string uniform = i.second;

            // Bind Correct Textures and Vertex Array Before Drawing
            glActiveTexture(GL_TEXTURE0 + unit);
            glBindTexture(GL_TEXTURE_2D, i.first);
            glUniform1i(glGetUniformLocation(shader, uniform.c_str()), unit++);
        }  glBindVertexArray(mVertexArray);
        glDrawArrays(GL_PATCHES, 0, mVertices.size());

        for (unsigned int i = 0; i < unit; ++i) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
    }

    size_t TessellationMesh::getObjectNum() {
        return mSubMeshes[0]->mIndices.size() / 3;
    }

    void TessellationMesh::setUnifromLOD(float lod) {
        mShader->activate().bind("tessellationLevel", lod);
    }
};
