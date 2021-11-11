#version 410 core  

in vec2 Tex_FS_in;  
in vec3 Pos_FS_in; 

uniform sampler2D diffuse;
uniform bool hasNoTexture;
uniform vec3 diffuseColor;
uniform vec3 gaze;
uniform bool showGaze;

layout(location = 0) out vec4 FragColor;  
layout(location = 1) out vec4 Gray; 
layout(location = 2) out vec4 Ecc; 

float d_c = 14804.6;
float r_m = 41.03;
vec2 a = vec2(0.9851, 0.9935);
vec2 r_2 = vec2(1.058, 1.035);
vec2 r_e = vec2(22.14, 16.35);
float square(float input)
{
  return pow(input,2.f);
}
float d_mf(float r, int k)
{
    return 2 * d_c / (1+r/r_m) * (pow((a[k]*(1+r/r_2[k])),-2) + (1-a[k])*exp(-r/r_e[k]));
}
float s_mf(float r, int k)
{
    return sqrt(2/(sqrt(3)*d_mf(r,k)));
}
float s_xy(float x, float y)
{
    float r_xy = length(vec2(x,y));
    if (r_xy == 0)
        return sqrt( square(s_mf(r_xy,0)) + square(s_mf(r_xy,1)));
    else
        return sqrt( square(x/r_xy*s_mf(r_xy,0)) + square(y/r_xy*s_mf(r_xy,1)) );
}

void main(){
	if (hasNoTexture) FragColor = vec4(diffuseColor, 1.0);
	else FragColor = vec4(texture(diffuse, Tex_FS_in.xy).xyz, 1.0);

	Gray.b = 0.299 * FragColor.r + 0.587 * FragColor.g + 0.114 * FragColor.b;

    vec3 fragDirection = normalize(Pos_FS_in);
    vec3 fragXDirection = normalize(vec3(fragDirection.x, 0.0, fragDirection.z));
    vec3 fragYDirection = normalize(vec3(0.0, fragDirection.y, fragDirection.z));
    vec3 gazeXDirection = normalize(vec3(gaze.x, 0.0, gaze.z));
    vec3 gazeYDirection = normalize(vec3(0.0, gaze.y, gaze.z));
    float Xdegree = degrees(acos(clamp(dot(gazeXDirection, fragXDirection), -1.0, 1.0)));
    float Ydegree = degrees(acos(clamp(dot(gazeYDirection, fragYDirection), -1.0, 1.0)));
    Ecc.r = 0.5 / s_xy(Xdegree, Ydegree);

    if (showGaze && dot(gaze, fragDirection) > 0.9998) {
        FragColor.xyz = vec3(0.0, 1.0, 0.0);
    }
}
