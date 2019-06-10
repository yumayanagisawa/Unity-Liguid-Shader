//Based on this shader https://www.shadertoy.com/view/XtBXzG on Shadertoy
Shader "Unlit/liquid"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
		iChannel0("iChannel0", Cube) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

			samplerCUBE iChannel0;

			#define SHADE 1

			static const float MAX_TRACE_DISTANCE = 10.0;
			static const float INTERSECTION_PRECISION = 0.001;
			static const int NUM_OF_TRACE_STEPS = 40;

			float sdSphere(float3 p, float radius) {
				return length(p) - radius;
			}

			float sdTorus(float3 p, float2 t)
			{
				float2 q = float2(length(p.xz) - t.x, p.y);
				return length(q) - t.y;
			}

			float3x3 calcLookAtMatrix(in float3 ro, in float3 ta, in float roll)
			{
				float3 ww = normalize(ta - ro);
				float3 uu = normalize(cross(ww, float3(sin(roll), cos(roll), 0.0)));
				float3 vv = normalize(cross(uu, ww));
				return float3x3(uu, vv, ww);
			}

			void doCamera(out float3 camPos, out float3 camTar, in float time, in float2 mouse) {
				float radius = 4.0;
				//float theta = 0.3 + 5.0*mouse.x - iTime * 0.15;
				float theta = 0.3 + 5.0 - _Time.y * .5;
				float phi = 3.14159*0.3;//mouse.y - iTime*0.5;

				float pos_x = radius * cos(theta) * sin(phi);
				float pos_z = radius * sin(theta) * sin(phi);
				float pos_y = radius * cos(phi);

				camPos = float3(pos_x, pos_y, pos_z);
				camTar = float3(0.0, -.5, 0.0);
			}

			float smin(float a, float b, float k) {
				float res = exp(-k * a) + exp(-k * b);
				return -log(res) / k;
			}

			float opRep(float3 p, float3 c)
			{
				float3 q = fmod(p, c) - 0.5*c;
				return sdSphere(q, .45);
			}

			float opScale(float3 p, float s)
			{
				return sdSphere(p / s, s)*s;
			}

			float3 opTwist(float3 p)
			{
				float  c = cos(10.0*p.y + 10.0);
				float  s = sin(10.0*p.y + 10.0);
				float2x2   m = float2x2(c, -s, s, c);
				float2 mm = mul(p.xz, m);
				return float3(mm, p.y);
			}

			float opS(float d1, float d2) {
				return max(-d1, d2);
			}

			float opU(float d1, float d2) {
				return min(d1, d2);
			}

			// checks to see which intersection is closer
			// and makes the y of the vec2 be the proper id
			float2 opU(float2 d1, float2 d2) {
				return (d1.x < d2.x) ? d1 : d2;
			}

			float opI(float d1, float d2) {
				return max(d1, d2);
			}

			// noise func
			float hash(float n) { return frac(sin(n)*753.5453123); }
			float noise(in float3 x)
			{
				float3 p = floor(x);
				float3 f = frac(x);
				f = f * f*(3.0 - 2.0*f);

				float n = p.x + p.y*157.0 + 113.0*p.z;
				return lerp(lerp(lerp(hash(n + 0.0), hash(n + 1.0), f.x),
					lerp(hash(n + 157.0), hash(n + 158.0), f.x), f.y),
					lerp(lerp(hash(n + 113.0), hash(n + 114.0), f.x),
						lerp(hash(n + 270.0), hash(n + 271.0), f.x), f.y), f.z);
			}

			float2 doModel(float3 pos) {


				float sphere = sdSphere(opTwist(pos), 1.75) + noise(pos * 1.25 + _Time.y * 1.2);
				//float sphere = opRep(pos+vec3(.8+iTime*.3, 0., 0.), vec3(.25)) + opScale(pos*.4, noise(pos+pos * .1 + iTime*0.25));//noise(pos * 1.5 + iTime*0.25);   
				float t1 = sphere;

				t1 = smin(t1, sdSphere(pos + float3(1.5, 1.0, 0.0), 0.4), 2.0);
				t1 = smin(t1, sdSphere(pos + float3(-1.8, 0.0, -1.0), 0.3), 2.0);
				t1 = smin(t1, sdSphere(pos + float3(1., 1.3, 1.3), 0.4), 3.0);

				return float2(t1, 1.0);
			}

			float shadow(in float3 ro, in float3 rd)
			{
				static const float k = 2.0;

				static const int maxSteps = 10;
				float t = 0.0;
				float res = 1.0;

				for (int i = 0; i < maxSteps; ++i) {

					float d = doModel(ro + rd * t).x;

					if (d < INTERSECTION_PRECISION) {

						return 0.0;
					}

					res = min(res, k*d / t);
					t += d;
				}

				return res;
			}

			float ambientOcclusion(in float3 ro, in float3 rd) {

				static const int maxSteps = 7;
				static const float stepSize = 0.05;

				float t = 0.0;
				float res = 0.0;

				// starting d
				float d0 = doModel(ro).x;

				for (int i = 0; i < maxSteps; ++i) {

					float d = doModel(ro + rd * t).x;
					float diff = max(d - d0, 0.0);

					res += diff;

					t += stepSize;
				}

				return res;
			}

			float3 calcNormal(in float3 pos) {

				float3 eps = float3(0.001, 0.0, 0.0);
				float3 nor = float3(
					doModel(pos + eps.xyy).x - doModel(pos - eps.xyy).x,
					doModel(pos + eps.yxy).x - doModel(pos - eps.yxy).x,
					doModel(pos + eps.yyx).x - doModel(pos - eps.yyx).x);
				return normalize(nor);
			}

			void doShading(float3 ro, float3 rd, inout float3 color, float3 currPos) {

				if (SHADE == 0) {
					float3 lightDir = normalize(float3(1.0, 0.4, 0.0));
					float3 normal = calcNormal(currPos);
					float3 normal_distorted = calcNormal(currPos + noise(currPos*1.5 + float3(0.0, 0.0, sin(_Time.y*0.75))));
					float shadowVal = shadow(currPos - rd * 0.1, lightDir);
					float ao = ambientOcclusion(currPos - normal * 0.01, normal);

					float ndotl = abs(dot(-rd, normal));
					float ndotl_distorted = abs(dot(-rd, normal_distorted));
					float rim = pow(1.0 - ndotl, 6.0);
					float rim_distorted = pow(1.0 - ndotl_distorted, 6.0);

					color = texCUBE(iChannel0, reflect(rd, normal)).xyz;
					color = lerp(color, float3(1., .7, .4), rim_distorted + 0.5);
					//color += rim;
					//color += texture (iChannel0, reflect(rd, normal_distorted)).xyz * rim + vec3(rim*.8);
					color += texCUBE(iChannel0, reflect(rd, normal_distorted)).xyz*rim + float3(rim*.8, rim*.8, rim*.8);


					color = lerp(color, float3(1., .7, .4), rim_distorted + 0.1);
					// color += rim;

					 //color *= vec3(mix(0.8,1.0,shadowVal));
					float aoCol = lerp(0.4, 1.0, ao);
					color *= float3(aoCol, aoCol, aoCol);
				}
				else {
					//------------------
					float3 lightDir = normalize(float3(0.0, -.4, 0.0));
					float3 normal = calcNormal(currPos - rd * noise(currPos*70.0)*0.1);

					//float shadowVal = shadow( currPos - rayDirection* 0.03, lightDir  );
					float ao = ambientOcclusion(currPos - normal * 0.01, normal);
					float ndotl = abs(dot(-rd, normal));
					float rim = pow(1.0 - ndotl, 3.5);
					float ndotl_distorted = abs(dot(-rd, normal));
					float rim_distorted = pow(1.0 - ndotl_distorted, 6.0);
					float specular = pow(dot(lightDir, normal), 3.0);

					float3 reflectionColor = texCUBE (iChannel0, reflect(rd, normal)).xyz;

					//color = vec3(0.2);
					color = reflectionColor;
					color *= float3(.7, 0.7, 0.7);
					color = lerp(color, float3(1., .7, .4), rim_distorted + 0.1);
					float aoCol = lerp(0.6, 1.0, ao);
					color *= float3(aoCol, aoCol, aoCol);
					color += specular * .5;
					color += rim * .5;

				}

			}

			float3 rayPlaneIntersection(float3 ro, float3 rd, float4 plane) {
				float t = -(dot(ro, plane.xyz) + plane.w) / dot(rd, plane.xyz);
				return ro + t * rd;
			}

			bool renderRayMarch(float3 ro, float3 rd, inout float3 color) {
				static const int maxSteps = NUM_OF_TRACE_STEPS;

				float t = 0.0;
				float d = 0.0;

				for (int i = 0; i < maxSteps; ++i) {
					float3 currPos = ro + rd * t;
					d = doModel(currPos).x;
					if (d < INTERSECTION_PRECISION) {
						break;
					}
					t += d;
				}

				if (d < INTERSECTION_PRECISION) {
					float3 currPos = ro + rd * t;
					doShading(ro, rd, color, currPos);

					return true;
				}

				float3 planePoint = rayPlaneIntersection(ro, rd, float4(0.0, 1.0, 0.0, 1.0));
				float shadowFloor = shadow(planePoint, float3(0.0, 1.0, 0.0));
				color = color * lerp(.9, 1.0, shadowFloor);

				return false;
			}

            fixed4 frag (v2f i) : SV_Target
            {
                
				float2 p = (-_ScreenParams.xy + 2.0*(_ScreenParams.xy * i.uv)) / _ScreenParams.y;
				//float2 p = i.uv;
				float2 m = (0,0);// iMouse.xy / iResolution.xy;

				// camera movement
				float3 ro, ta;
				doCamera(ro, ta, _Time.y, m);

				// camera matrix
				float3x3 camMat = calcLookAtMatrix(ro, ta, 0.0);  // 0.0 is the camera roll

				// create view ray
				float3 rd = normalize(mul(float3(p.xy, 1.25), camMat)); // 2.0 is the lens length

				// calc color
				float3 col = float3(0.8, .8, .85);
				renderRayMarch(ro, rd, col);

				// vignette, OF COURSE
				float vignette = 1.0 - smoothstep(1.0, 2.5, length(p));
				col.xyz *= lerp(0.7, 1.0, vignette);

				//fragColor = 
				return float4(col, 1.);
            }
            ENDCG
        }
    }
}
