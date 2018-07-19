//- Allegorithmic Metal/Rough and opacity PBR shader
//- ================================================
//-
//- Import from libraries.
import lib-alpha.glsl
import lib-pbr.glsl
import lib-pom.glsl

#define PBR_MOBILE 1

// Link Metal/Roughness MDL for Iray
//: metadata {
//:   "mdl":"mdl::alg::materials::physically_metallic_roughness::physically_metallic_roughness"
//: }

//- Show back faces as there may be holes in front faces.
//: state cull_face off

//- Channels needed for metal/rough workflow are bound here.
//: param auto channel_basecolor
uniform sampler2D basecolor_tex;
//: param auto channel_roughness
uniform sampler2D roughness_tex;
//: param auto channel_metallic
uniform sampler2D metallic_tex;
//: param auto channel_specularlevel
uniform sampler2D specularlevel_tex;


#if PBR_MOBILE
float rcp(float r)
{
	return 1.0 / r;
}

vec3 EnvBRDFApprox(vec3 SpecularColor, float Roughness, float NoV)
{
	// [ Lazarov 2013, "Getting More Physical in Call of Duty: Black Ops II" ]
	// Adaptation to fit our G term.
	const vec4 c0 = vec4(-1, -0.0275, -0.572, 0.022);
	const vec4 c1 = vec4(1, 0.0425, 1.04, -0.04);
	vec4 r = Roughness * c0 + c1;
	float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
	vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;

	return SpecularColor * AB.x + AB.y;
}
float PhongApprox(float Roughness, float RoL)
{
	float a = Roughness * Roughness;			// 1 mul
	//!! Ronin Hack?
	a = max(a, 0.008);						// avoid underflow in FP16, next sqr should be bigger than 6.1e-5
	float a2 = a * a;						// 1 mul
	float rcp_a2 = rcp(a2);					// 1 rcp
	//float rcp_a2 = exp2(-6.88886882 * Roughness + 6.88886882);

	// Spherical Gaussian approximation: pow( x, n ) ~= exp( (n + 0.775) * (x - 1) )
	// Phong: n = 0.5 / a2 - 0.5
	// 0.5 / ln(2), 0.275 / ln(2)
	float c = 0.72134752 * rcp_a2 + 0.39674113;	// 1 mad
	float p = rcp_a2 * exp2(c * RoL - c);		// 2 mad, 1 exp2, 1 mul
	// Total 7 instr
	return min(p, rcp_a2);						// Avoid overflow/underflow on Mali GPUs
}

float probabilityPhongApprox(float ndh, float vdh, float rol, float Roughness)
{
	return PhongApprox(Roughness, rol) * ndh / (4.0*vdh);
}
#endif

vec3 pbrComputeSpecularMobile(LocalVectors vectors, vec3 specColor, float glossiness, float occlusion)
{
  vec3 radiance = vec3(0.0);
  float roughness = 1.0 - glossiness;
  float ndv = dot(vectors.eye, vectors.normal);

#if PBR_MOBILE	
	specColor = EnvBRDFApprox(specColor, roughness, max(ndv, 0));
#endif

  for(int i=0; i<nbSamples; ++i)
  {
    vec2 Xi = fibonacci2D(i, nbSamples);
    vec3 Hn = importanceSampleGGX(
      Xi, vectors.tangent, vectors.bitangent, vectors.normal, roughness);
    vec3 Ln = -reflect(vectors.eye,Hn);

    float fade = horizonFading(dot(vectors.vertexNormal, Ln), horizonFade);

    float ndl = dot(vectors.normal, Ln);
    ndl = max( 1e-8, ndl );
    float vdh = max(1e-8, dot(vectors.eye, Hn));
    float ndh = max(1e-8, dot(vectors.normal, Hn));
#if PBR_MOBILE
	vec3 rfl = normalize(reflect(-vectors.eye, vectors.normal));
	float rdl = max(1e-8, dot(rfl, Ln));
    float lodS = roughness < 0.01 ? 0.0 : computeLOD(Ln, probabilityPhongApprox(ndh, vdh, rdl, roughness));
    radiance += fade * envSampleLOD(Ln, lodS) * specColor;
#else
    float lodS = roughness < 0.01 ? 0.0 : computeLOD(Ln, probabilityGGX(ndh, vdh, roughness));
    radiance += fade * envSampleLOD(Ln, lodS) *
      cook_torrance_contrib(vdh, ndh, ndl, ndv, specColor, roughness);
#endif
  }
  // Remove occlusions on shiny reflections
  radiance *= mix(occlusion, 1.0, glossiness * glossiness) / float(nbSamples);

  return radiance;
}

vec4 pbrComputeBRDFMobile(LocalVectors vectors, vec2 tex_coord, vec3 diffColor, vec3 specColor, float glossiness, float occlusion)
{
  return vec4(
      pbrComputeEmissive(emissive_tex, tex_coord)
	  + pbrComputeDiffuse(vectors.normal, diffColor, occlusion)
	  + pbrComputeSpecularMobile(vectors, specColor, glossiness, occlusion)
	  , 1.0);
}

vec4 pbrComputeBRDFMobile(V2F inputs, vec3 diffColor, vec3 specColor, float glossiness, float occlusion)
{
	LocalVectors vectors = computeLocalFrame(inputs);
	return pbrComputeBRDFMobile(vectors, inputs.tex_coord, diffColor, specColor, glossiness, occlusion);
}

//- Shader entry point.
vec4 shade(V2F inputs)
{
	// Apply parallax occlusion mapping if possible
	vec3 viewTS = worldSpaceToTangentSpace(getEyeVec(inputs.position), inputs);
	inputs.tex_coord += getParallaxOffset(inputs.tex_coord, viewTS);

	// Discard current fragment on the basis of the opacity channel
	// and a user defined threshold
	alphaKill(inputs.tex_coord);
  
	// Fetch material parameters, and conversion to the specular/glossiness model
	float glossiness = 1.0 - getRoughness(roughness_tex, inputs.tex_coord);
	vec3 baseColor = getBaseColor(basecolor_tex, inputs.tex_coord);
	float metallic = getMetallic(metallic_tex, inputs.tex_coord);
	float specularLevel = getSpecularLevel(specularlevel_tex, inputs.tex_coord);
#if PBR_MOBILE
	float dielectricSpec = 0.08 * specularLevel;
	vec3 diffColor = max(baseColor - baseColor * metallic, 0);		// 1 mad
	vec3 specColor = (dielectricSpec - dielectricSpec * metallic) + baseColor * metallic;	// 2 mad
#else
	vec3 diffColor = generateDiffuseColor(baseColor, metallic);
	vec3 specColor = generateSpecularColor(specularLevel, baseColor, metallic);
#endif
	// Get detail (ambient occlusion) and global (shadow) occlusion factors
	float occlusion = getAO(inputs.tex_coord) * getShadowFactor();

	// Feed parameters for a physically based BRDF integration
	return pbrComputeBRDFMobile(inputs, diffColor, specColor, glossiness, occlusion);
}

//- Entry point of the shadow pass.
void shadeShadow(V2F inputs)
{
}
