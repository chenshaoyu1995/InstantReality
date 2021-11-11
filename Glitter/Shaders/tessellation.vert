#version 410 core
layout (location = 0) in vec3 Pos_VS_in;
layout (location = 1) in vec3 Nor_VS_in;
layout (location = 2) in vec2 Tex_VS_in;
layout (location = 3) in vec3 Tan_VS_in;
layout (location = 4) in float lod_VS_in;
layout (location = 5) in float edgeLod_VS_in;

out vec3 Pos_CS_in;
out vec3 Nor_CS_in;
out vec2 Tex_CS_in;
out vec3 Tan_CS_in;
out int TriId_CS_in;
out float lod_CS_in;
out float edgeLod_CS_in;

void main()
{
	Pos_CS_in = Pos_VS_in;
	Tex_CS_in = Tex_VS_in;
	Nor_CS_in = Nor_VS_in;
	Tan_CS_in = Tan_VS_in;
	lod_CS_in = lod_VS_in;
	edgeLod_CS_in = edgeLod_VS_in;
	TriId_CS_in = gl_VertexID / 3;
}
