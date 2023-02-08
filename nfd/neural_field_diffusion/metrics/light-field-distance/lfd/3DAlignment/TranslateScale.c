#include <stdio.h>
#include <malloc.h>
#include <memory.h>
#include <math.h>
#include <float.h>

#include "ds.h"
#include "RWObj.h"

double max(double a, double b, double c)
{
	double d = (a>b)?a:b;
	return (c>d)?c:d;
}

void TranslateScale(pVer vertex, int NumVer, pTri triangle, int NumTri, char *fn, pVer T, double *S)
{
	Ver				Translate;	
	double			scale, dtmp;
//	FILE			*fpt;
//	char			filename[100];
	int				i, j, k;
	Ver				MinCoor, MaxCoor;

	// if vertices didn't use in any face, don't include it

	// get minimum and maximum coornidate from 3D model
	for(k=0; k<3; k++)
	{
		MinCoor.coor[k] = DBL_MAX;
		MaxCoor.coor[k] = -DBL_MAX;
	}
	for(i=0; i<NumTri; i++)
		for(j=0; j<triangle[i].NodeName; j++)
			for(k=0; k<3; k++)
			{
				dtmp = vertex[triangle[i].v[j]].coor[k];
				if( dtmp < MinCoor.coor[k] )
					MinCoor.coor[k] = dtmp;
				if( dtmp > MaxCoor.coor[k] )
					MaxCoor.coor[k] = dtmp;
			}		

	// get the translate and scale
	Translate.coor[0] = -( MinCoor.coor[0] + MaxCoor.coor[0] ) / 2;
	Translate.coor[1] = -( MinCoor.coor[1] + MaxCoor.coor[1] ) / 2;
	Translate.coor[2] = -( MinCoor.coor[2] + MaxCoor.coor[2] ) / 2;
	scale = 1.0 / max(	MaxCoor.coor[0]-MinCoor.coor[0], 
						MaxCoor.coor[1]-MinCoor.coor[1], 
						MaxCoor.coor[2]-MinCoor.coor[2]);

/*	// save the center and scale paramater
	sprintf(filename, "%s_TS.txt", fn);
	fpt = fopen(filename, "w");
	fprintf(fpt, "%f %f %f\n%.12f\n", Translate.coor[0], Translate.coor[1], Translate.coor[2], scale);
	fclose(fpt);
*/

	// translate and scale 3D model
	for(i=0; i<NumVer; i++)
		for(j=0; j<3; j++)
		{
			vertex[i].coor[j] += Translate.coor[j];
			vertex[i].coor[j] *= scale;
		}

	// return results
	T->coor[0] = Translate.coor[0];
	T->coor[1] = Translate.coor[1];
	T->coor[2] = Translate.coor[2];
	*S = scale;

//	sprintf(filename, "%s_ts.obj", srcfn);
//	SaveObj(filename, vertex, triangle, NumVer, NumTri);
}

void Translate(pVer vertex, int NumVer, Ver Translate)
{
	int			i, j;

	for(i=0; i<NumVer; i++)
		for(j=0; j<3; j++)
			vertex[i].coor[j] += Translate.coor[j];
}

void Scale(pVer vertex, int NumVer, double scale)
{
	int			i, j;

	for(i=0; i<NumVer; i++)
		for(j=0; j<3; j++)
			vertex[i].coor[j] *= scale;
}
