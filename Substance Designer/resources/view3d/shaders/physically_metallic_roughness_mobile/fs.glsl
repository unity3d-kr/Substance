//////////////////////////////// Fragment shader
#version 120
#extension GL_ARB_shader_texture_lod : require

#include "../common/common.glsl"
#include "../common/parallax.glsl"
#define PBR_MOBILE 1

varying vec3 iFS_Normal;
varying vec2 iFS_UV;
varying vec3 iFS_Tangent;
varying vec3 iFS_Binormal;
varying vec3 iFS_PointWS;

uniform vec3 Lamp0Pos = vec3(0.0,0.0,70.0);
uniform vec3 Lamp0Color = vec3(1.0,1.0,1.0);
uniform float Lamp0Intensity = 1.0;
uniform vec3 Lamp1Pos = vec3(70.0,0.0,0.0);
uniform vec3 Lamp1Color = vec3(0.198,0.198,0.198);
uniform float Lamp1Intensity = 1.0;

uniform float AmbiIntensity = 1.0;
uniform float EmissiveIntensity = 1.0;

uniform int parallax_mode = 0;

uniform float tiling = 1.0;
uniform vec3 uvwScale = vec3(1.0, 1.0, 1.0);
uniform bool uvwScaleEnabled = false;
uniform float envRotation = 0.0;
uniform float tessellationFactor = 4.0;
uniform float heightMapScale = 1.0;
uniform bool flipY = true;
uniform bool perFragBinormal = true;
uniform bool sRGBBaseColor = true;

uniform sampler2D heightMap;
uniform sampler2D normalMap;
uniform sampler2D baseColorMap;
uniform sampler2D metallicMap;
uniform sampler2D roughnessMap;
uniform sampler2D aoMap;
uniform sampler2D emissiveMap;
uniform sampler2D specularLevel;
uniform sampler2D opacityMap;
uniform sampler2D environmentMap;

uniform mat4 viewInverseMatrix;

// Number of miplevels in the envmap
uniform float maxLod = 12.0;

// Actual number of samples in the table
uniform int nbSamples = 16;
const int forcenbSamples = 4; //very low

// Irradiance spherical harmonics polynomial coefficients
// This is a color 2nd degree polynomial in (x,y,z), so it needs 10 coefficients
// for each color channel
uniform vec3 shCoefs[10];

// This must be included after the declaration of the uniform arrays since they
// can't be passed as functions parameters for performance reasons (on macs)
#include "../common/pbr_ibl.glsl"

float fit_roughness(float r)
{
// Fit roughness values to a more usable curve
	return r;
}

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

vec3 microfacets_brdf_Mobile(
	vec3 Nn,
	vec3 Ln,
	vec3 Vn,
	vec3 Ks,
	float Roughness)
{
	vec3 Hn = normalize(Vn + Ln);
	float vdh = max( 0.0, dot(Vn, Hn) );
	float ndh = max( 0.0, dot(Nn, Hn) );
	float ndl = max( 0.0, dot(Nn, Ln) );
	float ndv = max( 0.0, dot(Nn, Vn) );

	vec3 rfl = normalize(reflect(-Vn, Nn));
	float rol = max(0, dot(rfl, Ln));
		
#if PBR_MOBILE
	return fresnel(vdh,Ks) *
		( PhongApprox(Roughness, rol) * visibility(ndl,ndv,Roughness) / 4.0 );
#else
	return fresnel(vdh,Ks) *
		( normal_distrib(ndh,Roughness) * visibility(ndl,ndv,Roughness) / 4.0 );
#endif
}

vec3 pointLightContributionMobile(
	vec3 fixedNormalWS,
	vec3 pointToLightDirWS,
	vec3 pointToCameraDirWS,
	vec3 diffColor,
	vec3 specColor,
	float roughness,
	vec3 LampColor,
	float LampIntensity,
	float LampDist)
{
	// Note that the lamp intensity is using ˝computer games units" i.e. it needs
	// to be multiplied by M_PI.
	// Cf https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/

	return  max(dot(fixedNormalWS,pointToLightDirWS), 0.0) * ( (
		diffuse_brdf(
			fixedNormalWS,
			pointToLightDirWS,
			pointToCameraDirWS,
			diffColor*(vec3(1.0,1.0,1.0)-specColor))
		+ microfacets_brdf_Mobile(
			fixedNormalWS,
			pointToLightDirWS,
			pointToCameraDirWS,
			specColor,
			roughness) ) *LampColor*(lampAttenuation(LampDist)*LampIntensity*M_PI) );
}


vec3 IBLSpecularContributionMobile(
	sampler2D environmentMap,
	float envRotation,
	float maxLod,
	int nbSamples,
	vec3 normalWS,
	vec3 fixedNormalWS,
	vec3 Tp,
	vec3 Bp,
	vec3 pointToCameraDirWS,
	vec3 specColor,
	float roughness)
{
	vec3 sum = vec3(0.0,0.0,0.0);

	float ndv = max( 1e-8, abs(dot( pointToCameraDirWS, fixedNormalWS )) );

	for(int i=0; i<nbSamples; ++i)
	{
		vec2 Xi = fibonacci2D(i, nbSamples);
		vec3 Hn = importanceSampleGGX(Xi,Tp,Bp,fixedNormalWS,roughness);
		vec3 Ln = -reflect(pointToCameraDirWS,Hn);

		float ndl = dot(fixedNormalWS, Ln);

		// Horizon fading trick from http://marmosetco.tumblr.com/post/81245981087
		const float horizonFade = 1.3;
		float horiz = clamp( 1.0 + horizonFade * dot(normalWS, Ln), 0.0, 1.0 );
		horiz *= horiz;
		ndl = max( 1e-8, ndl );

		float vdh = max( 1e-8, abs(dot(pointToCameraDirWS, Hn)) );
		float ndh = max( 1e-8, abs(dot(fixedNormalWS, Hn)) );
#if PBR_MOBILE
		vec3 rfl = normalize(reflect(-pointToCameraDirWS, fixedNormalWS));
		float rol = max(0, dot(rfl, Ln));
		float lodS = roughness < 0.01 ? 0.0 :
			computeLOD(
				Ln,
				probabilityPhongApprox(ndh, vdh, rol, roughness),
				nbSamples,
				maxLod);
#else
		float lodS = roughness < 0.01 ? 0.0 :
			computeLOD(
				Ln,
				probabilityGGX(ndh, vdh, roughness),
				nbSamples,
				maxLod);
#endif
		sum += samplePanoramicLOD(environmentMap,rotate(Ln,envRotation),lodS) *
				microfacets_contrib(
				vdh, ndh, ndl, ndv,
				specColor,
				roughness) * horiz;
	}

	return sum / nbSamples;
}

vec3 computeIBLMobile(
	sampler2D environmentMap,
	float envRotation, 
	float maxLod,
	int nbSamples,
	vec3 normalWS,
	vec3 fixedNormalWS,
	vec3 iFS_Tangent,
	vec3 iFS_Binormal,
	vec3 pointToCameraDirWS,
	vec3 diffColor,
	vec3 specColor,
	float roughness,
	float ambientOcclusion)
{
	vec3 Tp,Bp;
	computeSamplingFrame(iFS_Tangent, iFS_Binormal, fixedNormalWS, Tp, Bp);

	vec3 result = IBLSpecularContributionMobile(
		environmentMap,
		envRotation,
		maxLod,
		nbSamples,
		normalWS,
		fixedNormalWS,
		Tp,
		Bp,
		pointToCameraDirWS,
		specColor,
		roughness);

	result += diffColor * (vec3(1.0,1.0,1.0)-specColor) *
		irradianceFromSH(rotate(fixedNormalWS,envRotation));

	return result * ambientOcclusion;
}



void main()
{
	vec3 normalWS = iFS_Normal;
	vec3 tangentWS = iFS_Tangent;
	vec3 binormalWS = perFragBinormal ?
		fixBinormal(normalWS,tangentWS,iFS_Binormal) : iFS_Binormal;

	vec3 cameraPosWS = viewInverseMatrix[3].xyz;
	vec3 pointToLight0DirWS = Lamp0Pos - iFS_PointWS;
	float pointToLight0Length = length(pointToLight0DirWS);
	pointToLight0DirWS *= 1.0 / pointToLight0Length;
	vec3 pointToLight1DirWS = Lamp1Pos - iFS_PointWS;
	float pointToLight1Length = length(Lamp1Pos - iFS_PointWS);
	pointToLight1DirWS *= 1.0 / pointToLight1Length;
	vec3 pointToCameraDirWS = normalize(cameraPosWS - iFS_PointWS);

	// ------------------------------------------
	// Parallax
	vec2 uvScale = vec2(1.0, 1.0);
	if (uvwScaleEnabled)
		uvScale = uvwScale.xy;
	vec2 uv = parallax_mode == 1 ? iFS_UV*tiling*uvScale : updateUV(
		heightMap,
		pointToCameraDirWS,
		normalWS, tangentWS, binormalWS,
		heightMapScale,
		iFS_UV,
		uvScale,
		tiling);

	// ------------------------------------------
	// Add Normal from normalMap
	vec3 fixedNormalWS = normalWS;  // HACK for empty normal textures
	vec3 normalTS = texture2D(normalMap,uv).xyz;
	if(length(normalTS)>0.0001)
	{
		normalTS = fixNormalSample(normalTS,flipY);
		fixedNormalWS = normalize(
			normalTS.x*tangentWS +
			normalTS.y*binormalWS +
			normalTS.z*normalWS );
	}

	// ------------------------------------------
	// Compute material model (diffuse, specular & roughness)
#if PBR_MOBILE
	float dielectricSpec = 0.08 * texture2D(specularLevel,uv).r;
	const float minRoughness=1e-4;
	
	vec3 baseColor = sRGBBaseColor ?
		srgb_to_linear(texture2D(baseColorMap,uv).rgb) :
		texture2D(baseColorMap,uv).rgb;
	float metallic = texture2D(metallicMap,uv).r;
	float roughness = max(minRoughness, fit_roughness(texture2D(roughnessMap,uv).r));
	
	vec3 diffColor = max(baseColor - baseColor * metallic, 0);		// 1 mad
	vec3 specColor = (dielectricSpec - dielectricSpec * metallic) + baseColor * metallic;	// 2 mad
	float NoV = max(dot(fixedNormalWS, pointToCameraDirWS), 0);
	specColor = EnvBRDFApprox(specColor, roughness, NoV);
#else
	float dielectricSpec = 0.08 * texture2D(specularLevel,uv).r;
	vec3 dielectricColor = vec3(dielectricSpec, dielectricSpec, dielectricSpec);
	const float minRoughness=1e-4;
	// Convert the base color from sRGB to linear (we should have done this when
	// loading the texture but there is no way to specify which colorspace is
	// uѕed for a given texture in Designer yet)
	vec3 baseColor = sRGBBaseColor ?
		srgb_to_linear(texture2D(baseColorMap,uv).rgb) :
		texture2D(baseColorMap,uv).rgb;
	float metallic = texture2D(metallicMap,uv).r;
	float roughness = max(minRoughness, fit_roughness(texture2D(roughnessMap,uv).r));

	vec3 diffColor = baseColor * (1.0 - metallic);
	vec3 specColor = mix(dielectricColor, baseColor, metallic);
#endif

	// ------------------------------------------
	// Compute point lights contributions
	vec3 contrib0 = pointLightContributionMobile(
			fixedNormalWS,
			pointToLight0DirWS,
			pointToCameraDirWS,
			diffColor,
			specColor,
			roughness,
			Lamp0Color,
			Lamp0Intensity,
			pointToLight0Length );
	vec3 contrib1 = pointLightContributionMobile(
			fixedNormalWS,
			pointToLight1DirWS,
			pointToCameraDirWS,
			diffColor,
			specColor,
			roughness,
			Lamp1Color,
			Lamp1Intensity,
			pointToLight1Length );

	// ------------------------------------------
	// Image based lighting contribution
	vec3 contribE = computeIBLMobile(
		environmentMap, envRotation, maxLod,
		forcenbSamples,
		normalWS, fixedNormalWS, tangentWS, binormalWS,
		pointToCameraDirWS,
		diffColor, specColor, roughness,
		AmbiIntensity * texture2D(aoMap,uv).r );

	// ------------------------------------------
	//Emissive
	vec3 emissiveContrib = srgb_to_linear(texture2D(emissiveMap,uv).rgb) * EmissiveIntensity;

	// ------------------------------------------
	vec3 finalColor = contrib0 + contrib1 + contribE + emissiveContrib;

	// Final Color
	// Convert the fragment color from linear to sRGB for display (we should
	// make the framebuffer use sRGB instead).
	gl_FragColor = vec4(finalColor, texture2D(opacityMap,uv));
}

