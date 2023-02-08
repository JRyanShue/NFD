#include <stdio.h>
#include <float.h>
#include <malloc.h>
#include "ds.h"
#include "RWObj.h"
#include "Rotate.h"

extern char srcfn[];
extern char destfn[];

double RecoverAffine(double **matrix, double cost[ANGLE][ANGLE][CAMNUM_2][CAMNUM_2], int *MinSrcCam)
{
	double		err, MinErr;
	int			align[60][20], i, j, k, angle, index, srcCam;
	FILE		*fpt;
	char		filename[100];
	pVer		VerRot;
	pTri		TriRot;
	int			NumVerRot, NumTriRot;
	vector		e1[2], e2[2];	// coordinate of edge

	// read align sequence
	fpt = fopen("align20.txt", "r");
	for(i=0; i<60; i++)
		for(j=0; j<CAMNUM_2; j++)
			fscanf(fpt, "%d", &align[i][j]);
	fclose(fpt);

	// get the minimum error among those alignment
	MinErr = DBL_MAX;
	for(srcCam=0; srcCam<ANGLE; srcCam++)	// each src angle
		for(i=0; i<ANGLE; i++)					// each dest angle
			for(j=0; j<60; j++)					// each align
			{
				err = 0;
				for(k=0; k<CAMNUM_2; k++)		// each vertex
					err += cost[srcCam][i][k][align[j][k]];

				if( err < MinErr )
				{
					MinErr = err;
					*MinSrcCam = srcCam;
					angle = i;
					index = j;
				}
			}

	sprintf(filename, "12_%1d", *MinSrcCam);
	ReadObj(filename, &VerRot, &TriRot, &NumVerRot, &NumTriRot);
	e1[0].x = VerRot[0].coor[0];
	e1[0].y = VerRot[0].coor[1];
	e1[0].z = VerRot[0].coor[2];
	e1[1].x = VerRot[1].coor[0];
	e1[1].y = VerRot[1].coor[1];
	e1[1].z = VerRot[1].coor[2];
	free(VerRot);
	free(TriRot);

	sprintf(filename, "12_%1d", angle);
	ReadObj(filename, &VerRot, &TriRot, &NumVerRot, &NumTriRot);
	e2[0].x = VerRot[align[index][0]].coor[0];
	e2[0].y = VerRot[align[index][0]].coor[1];
	e2[0].z = VerRot[align[index][0]].coor[2];
	e2[1].x = VerRot[align[index][1]].coor[0];
	e2[1].y = VerRot[align[index][1]].coor[1];
	e2[1].z = VerRot[align[index][1]].coor[2];
	free(VerRot);
	free(TriRot);

	RotateMatrix(matrix, e1, e2);

	// write the matrix to disk
	fpt=fopen("result.txt", "a");
	fprintf(fpt, "\n%s to %s\n", destfn, srcfn);
	fprintf(fpt, "SrcAngle = %d; DestAngle = %d; index = %d; err= %f\n", *MinSrcCam, angle, index, MinErr);
//	for(i=0; i<4; i++)
//	{
//		for(j=0; j<4; j++)
//			fprintf(fpt, "%lf ", matrix[i][j]);
//		fprintf(fpt, "\n");
//	}
	fclose(fpt);

	// return use which camera 
	return MinErr;
}

