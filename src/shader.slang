//import "shared.slang";

struct Vertex
{
    float3 pos;
    float3 color;
}

struct VertexData
{
    Vertex* verts;
    float time;
}

[[vk::push_constant]] VertexData DRAW_DATA;

#define TEXTURE_HEAP_BINDING 0
#define RWTEXTURE_HEAP_BINDING 1
#define SAMPLER_HEAP_BINDING 2

[[vk::binding(0, TEXTURE_HEAP_BINDING)]] public Texture2D texture2DHeap[];
[[vk::binding(0, RWTEXTURE_HEAP_BINDING)]] public RWTexture2D<float4> texture2DHeapRW[];
[[vk::binding(0, SAMPLER_HEAP_BINDING)]] public SamplerState samplerHeap[];

public SamplerState getSampler(uint samplerIndex)
{
    return samplerHeap[NonUniformResourceIndex(samplerIndex)];
}

public Texture2D getTexture2D(uint textureIndex)
{
    return texture2DHeap[NonUniformResourceIndex(textureIndex)];
}

struct VertexOutput
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
}

[shader("vertex")]
VertexOutput vertexMain(uint vert_id : SV_VertexID)
{
    let vertex = DRAW_DATA.verts[vert_id];//data.vert[vert_id];
    VertexOutput vert_out;
    //vert_out.pos = float4(float(vert_id%2), float(vert_id/2), 0.0, 1.0);
    //vert_out.color = float4(float3(float(vert_id)/4.), 1.0);
    vert_out.pos = float4(vertex.pos + float3(0., 0.*DRAW_DATA.time%1., 0.), 1.);
    //vert_out.color = float4(vertex.color, 1.);
    vert_out.uv = float2(float(vert_id%2), float(vert_id/2));
    return vert_out;
}

[shader("fragment")]
float4 fragmentMain(float2 uv : TEXCOORD0) : SV_Target
{
    return getTexture2D(1).Sample(getSampler(0), uv);
    //return color;
}