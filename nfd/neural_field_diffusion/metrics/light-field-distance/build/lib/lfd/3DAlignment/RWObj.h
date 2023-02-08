int ReadObj(char *filename, pVer *vertex, pTri *triangle, int *NumVer, int *NumTri);
void SaveObj(char *filename, pVer vertex, pTri triangle, int NumVer, int NumTri);
void SaveMergeObj(char *filename, pVer vertex1, pTri triangle1, int NumVer1, int NumTri1, 
								  pVer vertex2, pTri triangle2, int NumVer2, int NumTri2);
int ReadData(char *filename, pVer *vertex, pTri *triangle, int *NumVer, int *NumTri);
