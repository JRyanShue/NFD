#include <stdio.h>
#include <malloc.h>
#include <memory.h>

#include "ds.h"
#include "RWObj.h"
#include "RegionShape.h"
#include "Rotate.h"
#include "RWObj.h"
#include "Bitmap.h"
#include "TranslateScale.h"

#define	MAX_ITER	1

void RenderToMem(unsigned char *bmBits, pVer CamVertex, pVer v, pTri t, int nv, int nt);
extern int		winw, winh;
extern char		srcfn[];
extern char		destfn[];

double Distance(pVer CamVertex, unsigned char *destBuff[CAMNUM],
				double dest_Coeff[CAMNUM][ART_ANGULAR][ART_RADIAL],
				double src_Coeff[CAMNUM][ART_ANGULAR][ART_RADIAL],
				pVer vertex, pTri triangle, int NumVer, int NumTri)
{
	int				i;
	double			dist;
	double			CenX[CAMNUM], CenY[CAMNUM];

	for(i=0; i<CAMNUM; i++)
		RenderToMem(destBuff[i], CamVertex+i, vertex, triangle, NumVer, NumTri);

	FindRadius(destBuff, CenX, CenY);

	for(i=0; i<CAMNUM; i++)
		ExtractCoefficients(destBuff[i], dest_Coeff[i], NULL, CenX[i], CenY[i]);

	dist = 0;
	for(i=0; i<CAMNUM; i++)
		dist += GetDistance(dest_Coeff[i], src_Coeff[i]);

	return dist;
}

double Refine(double src_Coeff[CAMNUM][ART_ANGULAR][ART_RADIAL],
			  pVer vertex2, pTri triangle2, int NumVer2, int NumTri2, int UseCam,
			  pVer CamVertex, pTri CamTriangle, int CamNumVer, int CamNumTri)
{
	unsigned char	*destBuff[CAMNUM];	
	pVer			TmpVertex;
	pTri			TmpTriangle;
	int				TmpNumVer, TmpNumTri;		// total number of vertex and triangle.
	int				i, index=0, direct, flag, iter;
	double			angle;
	double			dist[3];
	char			filename[100];
	FILE			*fpt;
	char			word[]="XYZ";
	double			**matrix;
	vector			e1[2], e2[2];	// coordinate of edge
	// for region shape descriptor
	double			dest_Coeff[CAMNUM][ART_ANGULAR][ART_RADIAL];

	void			(*pRotate[3])(pVer, double , pVer , int );
	pRotate[0] = RotateX;
	pRotate[1] = RotateY;
	pRotate[2] = RotateZ;
	
	fpt = fopen("refine.txt", "a");
	fprintf(fpt, "\nDest: %s ; Src: %s\n", destfn, srcfn);

	// ********************************************************************************
	// capture CAMNUM silhouette of srcfn to memory
	sprintf(filename, "12_%d", UseCam);
	ReadObj(filename, &TmpVertex, &TmpTriangle, &TmpNumVer, &TmpNumTri);

	// ********************************************************************************
	// capture CAMNUM silhouette of destfn to memory,
	// and get the cost between srcfn silhouette and destfn silhouette
	// read REB only, so size is winw*winh
	for(i=0; i<CAMNUM; i++)
		destBuff[i] = (unsigned char *) malloc (winw * winh * sizeof(unsigned char));

	// initialize matrix and camera of model 1
	matrix = (double **) malloc (4 * sizeof(double *));
	for(i=0; i<4; i++)
	{
		matrix[i] = (double *) malloc(4 * sizeof(double));
		memset(matrix[i], 0, 4 * sizeof(double));
	}
	matrix[0][0] = matrix[1][1] = matrix[2][2] = matrix[3][3] = 1;
	e1[0].x = CamVertex[0].coor[0];
	e1[0].y = CamVertex[0].coor[1];
	e1[0].z = CamVertex[0].coor[2];
	e1[1].x = CamVertex[1].coor[0];
	e1[1].y = CamVertex[1].coor[1];
	e1[1].z = CamVertex[1].coor[2];

	// initialize dist[1]
	dist[1] = Distance(CamVertex, destBuff, dest_Coeff, src_Coeff, vertex2, triangle2, NumVer2, NumTri2);
	// iterative several times until no change
	iter = 0;
	do
	{
		flag = 0;
		iter ++;	// iterative times
		printf("Iterative %d\n", iter);
		fprintf(fpt, "Iterative %d\n", iter);

		// for each angle
		for(angle = 3.1415926 * 10.0 / 180.0; angle > 3.1415926 * 1.0 / 180.0; angle /= 2.0)
		{
			// Rotate x, y, z to each polyhedron
			for(direct=0; direct<3; direct++)
			{
				pRotate[direct](CamVertex, -angle, TmpVertex, CamNumVer);
				dist[0] = Distance(TmpVertex, destBuff, dest_Coeff, src_Coeff, vertex2, triangle2, NumVer2, NumTri2);
				pRotate[direct](CamVertex, angle, TmpVertex, CamNumVer);
				dist[2] = Distance(TmpVertex, destBuff, dest_Coeff, src_Coeff, vertex2, triangle2, NumVer2, NumTri2);

				if( dist[0] < dist[1] && dist[0] < dist[2])
				{	
					flag = 1;
					do
					{
						if( dist[1] < dist[0] )
							dist[0] = dist[1];

	//					pRotate[direct](vertex2, angle, vertex2, NumVer2);	
	printf("Rotate Camera %c: %f\tError: %f\n", word[direct], -angle, dist[0]);
	fprintf(fpt, "Rotate Camera %c: %f\tError: %f\n", word[direct], -angle, dist[0]);
	// save two model
	//sprintf(filename, "%s_to_%s_refine_%d.obj", destfn, srcfn, index++);
	//SaveMergeObj(filename, vertex1, triangle1, NumVer1, NumTri1, vertex2, triangle2, NumVer2, NumTri2);
						pRotate[direct](CamVertex, -angle, CamVertex, CamNumVer);
						pRotate[direct](CamVertex, -angle, TmpVertex, CamNumVer);
						dist[1] = Distance(TmpVertex, destBuff, dest_Coeff, src_Coeff, vertex2, triangle2, NumVer2, NumTri2);
					}while( dist[1] < dist[0] );
					dist[1] = dist[0];
				}
				else if( dist[2] < dist[1] && dist[2] < dist[0])
				{	
					flag = 1;
					do
					{
						if( dist[1] < dist[2] )
							dist[2] = dist[1];

	//					pRotate[direct](vertex2, -angle, vertex2, NumVer2);
	printf("Rotate Camera %c: %f\tError: %f\n", word[direct], angle, dist[2]);
	fprintf(fpt, "Rotate Camera %c: %f\tError: %f\n", word[direct], angle, dist[2]);
	// save two model
	//sprintf(filename, "%s_to_%s_refine_%d.obj", destfn, srcfn, index++);
	//SaveMergeObj(filename, vertex1, triangle1, NumVer1, NumTri1, vertex2, triangle2, NumVer2, NumTri2);
						pRotate[direct](CamVertex, angle, CamVertex, CamNumVer);
						pRotate[direct](CamVertex, angle, TmpVertex, CamNumVer);
						dist[1] = Distance(TmpVertex, destBuff, dest_Coeff, src_Coeff, vertex2, triangle2, NumVer2, NumTri2);
					}while( dist[1] < dist[2] );
					dist[1] = dist[2];
				}
			}

		}

	}while( flag && iter < MAX_ITER );

	e2[0].x = CamVertex[0].coor[0];
	e2[0].y = CamVertex[0].coor[1];
	e2[0].z = CamVertex[0].coor[2];
	e2[1].x = CamVertex[1].coor[0];
	e2[1].y = CamVertex[1].coor[1];
	e2[1].z = CamVertex[1].coor[2];
	RotateMatrix(matrix, e1, e2);
	Rotate(vertex2, NumVer2, matrix);

	for(i=0; i<4; i++)
		free(matrix[i]);
	free(matrix);
	free(TmpVertex);
	free(TmpTriangle);

	for(i=0; i<CAMNUM; i++)
		free(destBuff[i]);

	fclose(fpt);

	return dist[1];		// return the minimum error
}