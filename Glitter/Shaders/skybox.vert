#version 410 core
layout (location = 0) in vec3 Pos_VS_in;
layout (location = 1) in vec3 Nor_VS_in;
layout (location = 2) in vec2 Tex_VS_in;

uniform mat4 view;
uniform mat4 viewProjection; 

out vec2 Tex_FS_in; 
out vec3 Pos_FS_in; 

void main(){
	gl_Position = viewProjection * vec4(Pos_VS_in, 1.0);  
	Pos_FS_in  = (view * vec4(Pos_VS_in, 1.0)).xyz;
	Tex_FS_in = Tex_VS_in;
}
