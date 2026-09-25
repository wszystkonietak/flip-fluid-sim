#version 460 core
out vec4 FragColor;

in flat int VertexID; 
in float v_speedFactor;

void main()
{
	float dist = distance(vec2(0.5, 0.5), gl_PointCoord.xy);
	if (dist > 0.5)
		discard;

	vec3 slowColor = vec3(0.000, 0.322, 0.831);
	vec3 fastColor = vec3(0.000, 0.949, 0.996);

	vec3 finalColor = mix(slowColor, fastColor, v_speedFactor);

	FragColor = vec4(finalColor, 1.0);
}
