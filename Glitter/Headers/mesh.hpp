#pragma once
#include "shader.hpp"

// System Headers
#include <assimp/importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glad/glad.h>
#include <glm/glm.hpp>

// Standard Headers
#include <map>
#include <memory>
#include <vector>

// Define Namespace
namespace Mirage
{
	// Vertex Format
	struct Vertex {
		glm::vec3 position;
		glm::vec3 normal;
		glm::vec2 uv;
	};

	class Mesh
	{
	public:

		// Implement Default Constructor and Destructor
		 Mesh() { glGenVertexArrays(1, & mVertexArray); }
		~Mesh() { glDeleteVertexArrays(1, & mVertexArray); }

		// Implement Custom Constructors
		Mesh(std::string const & filename);
		Mesh(std::vector<Vertex> vertices,
			 std::vector<GLuint> indices,
			 std::map<GLuint, std::string> textures);

		// Public Member Functions
		virtual void draw(GLuint shader) = 0;
		virtual size_t getObjectNum() { return 1; };
		virtual void setUnifromLOD(float lod) {};

		// Public Member Containers
		std::vector<std::shared_ptr<Mesh>> mSubMeshes;
		std::vector<GLuint> mIndices;
		std::vector<Vertex> mVertices;

	protected:

		// Disable Copying and Assignment
		Mesh(Mesh const &) = delete;
		Mesh & operator=(Mesh const &) = delete;

		// Private Member Functions
		bool isTexture16(std::string filename);
		GLuint loadTexture(std::string filename);
		GLuint loadTexture16(std::string filename);
		GLuint loadAiTexture(const aiTexture*);
		std::map<GLuint, std::string> process(std::string const& path, aiMaterial* material, aiTextureType type, aiScene const* scene);

		void parseNode(std::string const& path, aiNode const* node, aiScene const* scene);
		virtual void parse(std::string const& path, aiMesh const* mesh, aiScene const* scene) = 0;

		// Private Member Variables
		GLuint mVertexArray;
		GLuint mVertexBuffer;
		GLuint mElementBuffer;
		std::map<GLuint, std::string> mTextures;
	};

	class TexMesh : public Mesh {
	public:
		TexMesh(std::vector<std::shared_ptr<Mesh>>& meshes) { mSubMeshes = meshes; };
		TexMesh(std::string const& filename);
		TexMesh(std::vector<Vertex> vertices, std::vector<GLuint> indices, std::map<GLuint, std::string> textures);
		virtual void draw(GLuint shader);

		bool mNoTexture = false;
		glm::vec3 mDiffuseColor;

	protected:
		virtual void parse(std::string const& path, aiMesh const* mesh, aiScene const* scene);
	};

	class TessellationMesh : public Mesh {
	public:
		TessellationMesh(std::string const& filename);
		TessellationMesh(std::vector<Vertex>vertices, std::vector<GLuint> indices, std::map<GLuint, std::string> textures, std::vector<glm::vec3> tangents, size_t vertSize);
		void SetupTesBuffer(GLuint faceTesBuffer, GLuint edgeTesBuffer);
		virtual void draw(GLuint shader);
		virtual size_t getObjectNum();
		virtual void setUnifromLOD(float lod);

		Shader* mShader;
	protected:
		virtual void parse(std::string const& path, aiMesh const* mesh, aiScene const* scene);

		GLuint mFaceTesBuffer = 0;
		GLuint mEdgeTesBuffer = 0;
		GLuint mTangentBuffer = 0;
		size_t mVertSize;
		std::vector<glm::vec3> mTangents;
	};
};
