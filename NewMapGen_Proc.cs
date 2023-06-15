
using System;
using CN_Noise;
using Sirenix.OdinInspector;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
public class NewMapGen_Proc : MonoBehaviour
{
    //Noise
    private CustomNoise HeightMap_Noise = new CustomNoise();
    
    [Header("Mesh Data")] 
    public MeshFilter Filter;
    [Space]
    public NewDataMesh Mesh_Data;
    [Header("Noise Alteration")]
    public AnimationCurve Curve; 
    public nativeCurve NativeCurve; 
    [Header("Noise")]
    public Noise_Data noise_Data;
    
    public Type_Terrain[] Regions;
    
    private float[] HeightMap;
    private Color32[] ColorMap;

    private Chunk NoiseChunk;

    private NativeArray<Type_Terrain> NativeRegions;
 

   #region Structs
   
   #region MeshData
   [Serializable]
   public struct NewDataMesh
   {
       [Range(1,255)]
       public int SetSizeX, SetSizeZ;

       [Space] 
       public float Set_XSpacing;
       public float Set_ZSpacing;
       
       public float XSpacing => Set_XSpacing / 255;   // to ensure the size of the map is 1:1 scale
       public float ZSpacing => Set_ZSpacing / 255;   // to ensure the size of the map is 1:1 scale

       
       public float Width; // width of the mesh 
       [Space, Range(1, 6)] 
       public int Lod;

       public int SizeX => SetSizeX - 1;
       public int SizeZ => SetSizeZ - 1;

       //NON LOD 
       public int TotalVerts => (SizeX + 1) * (SizeZ + 1); // total amounts of verts for mesh 
       public int TotalTris => (SizeX ) * (SizeZ ) * 6; // total amount of tris for mesh
       public int MeshJobIterations => SizeX * SizeZ; // total amount of iterations for meshGenJob 
       
       public float TileLength => Width / SizeX;

       //LOD Version
       
       public int LOD_TotalVerts => ((SizeX/Lod) + 1) * ((SizeZ/Lod) + 1); // total amounts of verts for mesh 
       public int LOD_TotalTris => ((SizeX/Lod) ) * ((SizeZ/Lod) ) * 6; // total amount of tris for mesh
       public int LOD_MeshJobIterations => (SizeX/Lod) * (SizeZ/Lod); // total amount of iterations for meshGenJob 
      
       public float LOD_TileLength => Width / (SizeX/Lod);
   }
   
   #endregion
   
   #region noise Data

   [Serializable]
   public struct Noise_Data
   {
       public float NoiseMultiplier; // multiply noise by this value
       [Space]
       public int NoiseWidth;
       public int NoiseHeight;
        
       [Space]
       public int octaves;
       public float Lacunarity;
       public float Gain;

       public float Scale;
       public float Frequency;
       public Vector3 Offset;
       public Vector3 DivOffset;
       public string Seed;
       [Space] 
       public float FallOff_Steepness;
       public float FallOff_Offset;
       public bool UseFalloff;
       [Space] 
       public CustomNoise.NoiseType NoiseType;
       public CustomNoise.CellularDistanceFunction DistanceFunction;
       public CustomNoise.FractalType FractalType;
       public CustomNoise.Interp InterpolationType;
       [Space]
       public float minNoiseValue;
       public float maxNoiseValue;
   }

   #endregion
   
   #region Regions_Data _colour
   
   [Serializable]
   public struct Type_Terrain
   {
       public float height;
       public Color32 colour;
   }
   
   #endregion

   #region Chunks
   
   public struct Chunk
   {
       [NativeDisableParallelForRestriction] public NativeArray<float> NewHeightMap;
       // [NativeDisableParallelForRestriction] public NativeArray<float> OriginalHeightMap;
       [NativeDisableParallelForRestriction] public NativeArray<Color32> ColorMap;
       public readonly float minvalue;
       public readonly float maxvalue;

       public Chunk(NativeArray<float> h) //, NativeArray<Color> c)
       {
           minvalue = 100;
           maxvalue = -100;
           for (int i = 0; i < h.Length; i++)
           {
               if (h[i] < minvalue)
               {
                   minvalue = h[i];
               }
               else if (h[i] > maxvalue)
               {
                   maxvalue = h[i];
               }
           }

           NewHeightMap = h;
           //   OriginalHeightMap = h;
           ColorMap = new NativeArray<Color32>(h.Length,Allocator.TempJob);
           //   ColorMap = c;
       }
    
       public void Dispose()
       { 
           ColorMap.Dispose();
           NewHeightMap.Dispose();
           //    OriginalHeightMap.Dispose();
       }
   }

   #endregion
   #endregion
   
   #region Jobs
   
   #region MeshJob
   
   #region CreatePlaneMesh
   
   public struct GenerateMesh : IJobParallelFor
   {
       [NativeDisableParallelForRestriction] private NativeArray<Vector3> Verticies;
       [NativeDisableParallelForRestriction] private NativeArray<int> Triangles;
       [NativeDisableParallelForRestriction] private NativeArray<Vector2> uvs;
       private readonly float Height;
       private readonly NewDataMesh MeshData;
       private readonly Chunk chunk_Data;

       public GenerateMesh(NewDataMesh meshdata, Chunk map, float height)
       {
           Verticies = new NativeArray<Vector3>(meshdata.TotalVerts, Allocator.TempJob);
           Triangles = new NativeArray<int>(meshdata.TotalTris, Allocator.TempJob);
           uvs = new NativeArray<Vector2>(meshdata.TotalVerts, Allocator.TempJob);
           chunk_Data = map;
           MeshData = meshdata;
           Height = height;
       }
       
       public void Execute(int index)
       {
           int y = index % MeshData.SizeX;
           int x = index / MeshData.SizeX;
           
            //have to calculate all 4 indices;
            //calculate bottom left to top right
            
            //bottom left
            int Bl = index + Mathf.FloorToInt(index / MeshData.SizeX); //bottom left
            //BottomRight
            int Br = Bl + 1;
            //TopLeft
            int Tl = Br+ MeshData.SizeX;
            //TopRight
            int Tr = Tl + 1;
            
            //xspacing = horizontal spacing between each point , increase this if using lod to remove points
            //zspacing = Vertical spacing between each point , increase this if using lod to remove points

            Vector3 CenterOffset = new Vector3((MeshData.SizeX * MeshData.XSpacing) / 2f, 0f, (MeshData.SizeZ * MeshData.ZSpacing) / 2f);
            
            
            //Set vertices;

            Verticies[Bl] = (new Vector3(x * MeshData.XSpacing, chunk_Data.NewHeightMap[Bl]* Height, (y) * MeshData.ZSpacing) - CenterOffset) * MeshData.TileLength;
            Verticies[Br] = (new Vector3((x) * MeshData.XSpacing, chunk_Data.NewHeightMap[Br]* Height, (y + 1) * MeshData.ZSpacing) - CenterOffset) * MeshData.TileLength;
            Verticies[Tl] = (new Vector3((x + 1) * MeshData.XSpacing, chunk_Data.NewHeightMap[Tl]* Height, (y) * MeshData.ZSpacing) - CenterOffset) * MeshData.TileLength;
            Verticies[Tr] = (new Vector3((x + 1) * MeshData.XSpacing, chunk_Data.NewHeightMap[Tr]* Height, (y + 1) * MeshData.ZSpacing) - CenterOffset) * MeshData.TileLength;
            
            
            uvs[Bl] = new Vector2((float)(x) / MeshData.SizeX, (float)(y) / MeshData.SizeZ);
            uvs[Br] = new Vector2((float)(x) / MeshData.SizeX, (float)(y+1) / MeshData.SizeZ);
            uvs[Tl] = new Vector2((float)(x+1) / MeshData.SizeX, (float)(y) / MeshData.SizeZ);
            uvs[Tr] = new Vector2((float)(x+1) / MeshData.SizeX, (float)(y+1) / MeshData.SizeZ);


           //triangle index = index *6

            //triangle position =  0, 1, 2, 1 , 3 , 2 // 
            // c - - - - d
            // |         |
            // |         |
            // |         |
            // a - - - - b
            // a is bottom left and the rest of the points are calculated using the index of a
            // we are only looping through each square to calculate the triangle and other bs
            
            //starts from index 0 to resolution index
            //Polygon 1 
            Triangles[index * 6 ] = Bl;    //set bottom left for polygon as vertex Bl
            Triangles[index * 6 + 1] = Br; // set bottom right for polygon as vertex Br
            Triangles[index * 6 + 2] = Tl; // set Top left for polygon as vertex Tl
            //Polygon2 
            Triangles[index * 6 + 3] = Br; // set bottom right for polygon as vertex Br
            Triangles[index * 6 + 4] = Tr; // set Top Right for polygon as vertex Tr
            Triangles[index * 6 + 5] = Tl; // set Top Left for polygon as vertex Tl

        }

        public Vector3[] GetVerts()
        {
            return Verticies.ToArray(); ///return vertices array as a managed array
        }

        public void Dispose()
        {
            Verticies.Dispose(); //dispose of vertices native array
            Triangles.Dispose(); // dispose of triangles native array
            uvs.Dispose(); // dispose of uvs native array
        }

        public Mesh GetMesh()
        {
            Mesh m = new Mesh(); // create a new mesh 

            m.SetVertices(Verticies); // set the vertices
            m.triangles = Triangles.ToArray(); // set the triangles
            m.uv = uvs.ToArray(); // set the uvs
            m.RecalculateBounds(); // recalculate the map bounds
            m.RecalculateNormals(); // recalculate the map normals
            Dispose(); // Dispose of the Native Array
            m.SetColors(chunk_Data.ColorMap); //set colours of vertices

            return m;
        }

   }

   #endregion
   
   #endregion
   
   #region NoiseJob

   #region GenerateNoise

   public struct GenerateNoise : IJobParallelFor
   {
       [NativeDisableParallelForRestriction] private NativeArray<float> height;
       private readonly int Width;
       private readonly int Height;
       private readonly CustomNoise noise;
       private readonly int WSize;
       private readonly int VSize;

       public GenerateNoise(CustomNoise newNoise , int WidthSize,int VerticalSize ,int height, int width)
       {
           this.height = new NativeArray<float>((WidthSize +1) * (VerticalSize + 1),Allocator.TempJob);
           noise = newNoise;
           Width = width;
           WSize = WidthSize;
           VSize = VerticalSize;
           this.Height = height;
       }

       public void Execute(int index)
       {
           //create a square noise map 
           int x = index / (WSize+1);  //set x to be dependant on the current index
           int y = index % (WSize+1); // set y to be dependant on the current index
           height[index] = noise.GetNoise(x, y,Width,Height);
       }

       public Chunk GetChunk => new Chunk(height);

   }
   

   #endregion
   #region AlterNoise

   public struct AlterNoise : IJobParallelFor
   {
       [NativeDisableParallelForRestriction] private Chunk val;
       [NativeDisableParallelForRestriction] private readonly NativeArray<Type_Terrain> regions;
       [NativeDisableParallelForRestriction] private readonly nativeCurve curve;
       private readonly float height;

       public AlterNoise(ref Chunk map, NativeArray<Type_Terrain> region, float Height, nativeCurve nCurve)
       {
           val = map;
           regions = region;
           this.height = Height;
           curve = nCurve;
       }

       public void Execute(int index)
       {
           //Inverse lerp the noise value between 0 and 1;
           val.NewHeightMap[index] = math.unlerp(val.minvalue, val.maxvalue, val.NewHeightMap[index]);

           //minvalue for colour
           float minvalue = 10000;
           
           // Set Colour Based on preset Regions
           for (int i = 0; i < regions.Length; i++)
           {
               if (regions[i].height < minvalue && val.NewHeightMap[index] < regions[i].height)
               {
                   minvalue = regions[i].height;
                   val.ColorMap[index] = regions[i].colour;
               }
           }

           // Map the height onto the curve
           val.NewHeightMap[index] = curve.Evaluate(val.NewHeightMap[index]);
           // multiply the height by the height multiplier;
           val.NewHeightMap[index] *= height;


       }
       
       public void Dispose()
       {
           regions.Dispose();
       }
   }

   #endregion
   
   #endregion
   
   
   #endregion
   
   #region Methods
   #region InitializeNoise
    
   public void InitializeNoise(ref CustomNoise noise, Noise_Data data)
   {
       noise.SetMinHeight(data.minNoiseValue);
       noise.SetMaxHeight(data.maxNoiseValue);
        
       noise.SetNoiseType(data.NoiseType);
       noise.SetCellularDistanceFunction(data.DistanceFunction);
       noise.SetFractalType(data.FractalType);
       noise.SetInterp(data.InterpolationType);
      
       noise.setTextureWidth(data.NoiseWidth );
       noise.setTextureHeight(data.NoiseHeight );
       noise.SetSeed(data.Seed.GetHashCode());
       noise.SetFractalLacunarity(data.Lacunarity);
       noise.SetFractalGain(data.Gain);
       noise.SetScale(data.Scale);
       noise.SetFractalOctaves(data.octaves);
       noise.SetFrequency(data.Frequency);
       noise.SetOffset(data.Offset);
       noise.SetDivOffset(data.DivOffset);
      
       noise.SetFallOffOffset(data.FallOff_Offset);
       noise.SetFallOffSteepness(data.FallOff_Steepness);
       noise.SetUseFallOff(data.UseFalloff);  
   }
   #endregion

   //todo make it so the x and z scale so same size irrelevant of how many x size or y size 
   private void OnValidate()
   {
       MainUpdate();
   }

   [Button]
   private void MainUpdate()
   {
       InitializeNoise(ref HeightMap_Noise, noise_Data);
       
       //Initialize Regions
        
       // Allocate memory for the native regions array
       NativeRegions = new NativeArray<Type_Terrain>(Regions.Length,Allocator.TempJob);

       // Copy the elements from the managed array to the native array
       NativeRegions.CopyFrom(Regions);

       //Native curve initialize 
       if (Curve != null )
       {
           NativeCurve.Update(Curve,32);
       }
       
       CreateChunk();
   }

   void CreateChunk()
   {
       
       // ===========================================  GENERATE NOISE =================================================================
       //Generate Noise
       GenerateNoise NoiseJob = new GenerateNoise(HeightMap_Noise, Mesh_Data.SizeX,Mesh_Data.SizeZ,noise_Data.NoiseHeight,noise_Data.NoiseWidth);
       NoiseJob.Schedule(Mesh_Data.TotalVerts, 10000).Complete(); // start job, then wait till job is complete
       NoiseChunk = NoiseJob.GetChunk;
       //Alter Values of NoiseMap;
       AlterNoise AlterringJob = new AlterNoise(ref NoiseChunk, NativeRegions, noise_Data.NoiseMultiplier, NativeCurve);
       AlterringJob.Schedule(Mesh_Data.TotalVerts,10000).Complete();
       AlterringJob.Dispose();

       HeightMap = NoiseChunk.NewHeightMap.ToArray();
       ColorMap = NoiseChunk.ColorMap.ToArray();
       
       // ==============================================================================================================================
       // ===========================================  GENERATE MESH  ==================================================================
       GenerateMesh MeshGen = new GenerateMesh(Mesh_Data, NoiseChunk, noise_Data.NoiseMultiplier);
       MeshGen.Schedule(Mesh_Data.MeshJobIterations,10000).Complete();
       
       Filter.mesh = MeshGen.GetMesh();
       NoiseChunk.Dispose();
       // MeshGen.Dispose();
   }

   #endregion

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
