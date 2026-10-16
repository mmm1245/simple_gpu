import "shared.slang";

struct Vertex
{
    float3 pos;
    float3 color;
}

struct VertexData
{
    Vertex* verts;
}

struct VertexOutput
{
    float4 pos : SV_Position;
    float4 color : TEXCOORD0;
}

[shader("vertex")]
VertexOutput vertexMain(uint vert_id : SV_VertexID, GraphicsPipelineData<VertexData> data)
{
    VertexOutput vert_out;
    vert_out.pos = float4(float(vert_id%2), float(vert_id/2), 0.0, 1.0);
    vert_out.color = float4(float3(1.0), 1.0);
    return vert_out;
}

[shader("fragment")]
float4 fragmentMain(float4 color : TEXCOORD0) : SV_Target
{
    return color;
}