#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>
#include "ds.h"

#define		LINE_MAX_LEN	256

int ReadMeterial(char *filename, pMeterial *color)
{
	FILE		*fpt;
	char		input[LINE_MAX_LEN];
	char		token[LINE_MAX_LEN];
	char		*next;
	int			width;
	pMeterial	pNewMtl;

	// if cannot find the mtl file
	//  Advance to the first nonspace character in "input"
	if( (fpt = fopen(filename, "r")) == NULL )
		return 0;

	while( fgets(input, LINE_MAX_LEN, fpt) != NULL ) 
	{
		//  Advance to the first nonspace character in "input"
		for ( next = input; *next != '\0' && isspace(*next); next++ )
			;

		// Skip blank lines and comments
		if ( *next == '\0' || *next == '#' || *next == '$')
			continue;

		// Extract the first word in this line. 
		sscanf ( next, "%s%n", token, &width );
		next = next + width;

		if( strcmp(token, "newmtl") == 0 )
		{
			pNewMtl = (pMeterial) malloc(sizeof(Meterial));
			sscanf(next, "%s", pNewMtl->name);
			pNewMtl->pointer = (*color);
			(*color) = pNewMtl;
		}
		else if( strcmp(token, "Kd") == 0 )
		{
			sscanf(next, "%lf %lf %lf", &(pNewMtl->r), &(pNewMtl->g), &(pNewMtl->b));
		}
	}

	fclose(fpt);
	return 1;
}

int ReadObj(char *filename, pVer *vertex, pTri *triangle, int *NumVer, int *NumTri)
{
	FILE		*fpt;
	char		input[LINE_MAX_LEN];
	char		token[LINE_MAX_LEN];
	char		*next;
	int			width;
	int			numver, numtri;
	int			VerIndex, TriIndex;
	double		r0, r1, r2;
	char		value[LINE_MAX_LEN];
	// the face is triangle, but sometimes, the face has 4 vertex, split it to two triangles
	int			f[50];	// the maximum vertices of a face is defined as 50 ( 3 is normal )
	int			fn;		// vertex number of a face
	int			i;
	// for color
	pMeterial	color = NULL;
	pMeterial	pNowMtl = NULL;
	char			fname[400];
	
	sprintf(fname, "%s.obj", filename);
	if( (fpt = fopen(fname, "r")) == NULL )
		return 0;	// False

	// one pass: get the number of vertex and triangle
	numver = 0;
	numtri = 0;
	color = NULL;
	while( fgets(input, LINE_MAX_LEN, fpt) != NULL ) 
	{

		//  Advance to the first nonspace character in "input"
		for ( next = input; *next != '\0' && isspace(*next); next++ )
			;

		// Skip blank lines and comments
		if ( *next == '\0' || *next == '#' || *next == '$')
			continue;

		// Extract the first word in this line. 
		sscanf ( next, "%s%n", token, &width );
		next = next + width;

		if( strcmp(token, "v") == 0 )
			numver ++;
		else if( strcmp(token, "f") == 0 )
			numtri ++;
		// read the color
		else if( strcmp(token, "mtllib") == 0 )
		{
			sscanf ( next, "%s", token );
			if( !ReadMeterial(token, &color) )
				;//return 0;
		}
	}
	*NumVer = numver;
	*NumTri = numtri;

	// allocate memory of vertex and triangle
	*vertex = (pVer) malloc( numver * sizeof(Ver));
	memset(*vertex, 0, numver * sizeof(Ver));
	*triangle = (pTri) malloc( numtri * sizeof(Tri));
	memset(*triangle, 0, numtri * sizeof(Tri));

	// two pass: get data of vertex and triangle
	fseek(fpt, 0, SEEK_SET);

	VerIndex = 0;
	TriIndex = 0;
	pNowMtl = NULL;
	while( fgets(input, LINE_MAX_LEN, fpt) != NULL )
	{

		//  Advance to the first nonspace character in INPUT
		for ( next = input; *next != '\0' && isspace(*next); next++ )
			;

		// Skip blank lines and comments
		if ( *next == '\0' || *next == '#' || *next == '$')
			continue;

		// Extract the first word in this line. 
		sscanf ( next, "%s%n", token, &width );
		
		// Set NEXT to point to just after this token. 
		next = next + width;

		/*	V X Y Z W
			Geometric vertex.
			W is optional, a weight for rational curves and surfaces.
			The default for W is 1.		*/
		if( strcmp(token, "v") == 0 )
		{
			sscanf ( next, "%lf %lf %lf", &r0, &r1, &r2 );

			(*vertex)[VerIndex].coor[0] = r0;
			(*vertex)[VerIndex].coor[1] = r1;
			(*vertex)[VerIndex].coor[2] = r2;

			VerIndex ++;
		}
		/*  F V1 V2 V3
			or
			F V1/VT1/VN1 V2/VT2/VN2 ...
			or
			F V1//VN1 V2//VN2 ...
			or
			F V1/VT1/ V2/VT2/ ...

			Face.
			A face is defined by the vertices.
			Optionally, slashes may be used to include the texture vertex
			and vertex normal indices.

			OBJ line node indices are 1 based rather than 0 based.
			So we have to decrement them before loading them into FACE.			*/
		// sometimes in a obj file, there maybe use two kind mode of "f" in different group
		// so, it have to check format when starting read "f"
		else if( strcmp(token, "f") == 0 )
		{
			fn = 0;
			while( sscanf( next, "%s%n", value, &width) != EOF )
			{
				sscanf( value, "%d", f+fn);
				next = next + width;
				fn ++;
			}

			for(i=0; i<fn; i++)
				(*triangle)[TriIndex].v[i] = f[i] - 1;

			(*triangle)[TriIndex].NodeName = fn;	// record the number of vertex in this triangle

			// color
			if( pNowMtl )
			{
				(*triangle)[TriIndex].r = pNowMtl->r;
				(*triangle)[TriIndex].g = pNowMtl->g;
				(*triangle)[TriIndex].b = pNowMtl->b;
			}

			TriIndex ++;
		}
		else if( strcmp(token, "usemtl") == 0 )
		{
			sscanf(next, "%s", token);
			for(pNowMtl=color; pNowMtl && strcmp(pNowMtl->name, token); pNowMtl=pNowMtl->pointer)
				;
		}
	}

	// free memory of meterial
	for(pNowMtl=color; pNowMtl; pNowMtl=color)
	{
		color = color->pointer;
		free(pNowMtl);
	}

	fclose(fpt);
	return 1;	// True
}

void SaveObj(char *filename, pVer vertex, pTri triangle, int NumVer, int NumTri)
{
	int		i, j;
	FILE	*fpt;

	fpt = fopen(filename, "w");

	// save each vertex to .obj file
	for(i=0; i<NumVer; i++)
		fprintf(fpt, "v %.6f %.6f %.6f\n", vertex[i].coor[0], vertex[i].coor[1], vertex[i].coor[2]);

	// save each triangle (face)
	for(i=0; i<NumTri; i++)
	{
		fprintf(fpt, "f");
		for(j=0; j<triangle[i].NodeName; j++)	// record the number of vertex in this triangle
			fprintf(fpt, " %d", triangle[i].v[j]+1);
		fprintf(fpt, "\n");
	}

	fclose(fpt);
}

void SaveMergeObj(char *filename, pVer vertex1, pTri triangle1, int NumVer1, int NumTri1, 
								  pVer vertex2, pTri triangle2, int NumVer2, int NumTri2)
{
	int		i, j;
	FILE	*fpt;

	fpt = fopen(filename, "w");

	fprintf(fpt, "# src model: %d vertices; %d triangles\n", NumVer1, NumTri1);
	fprintf(fpt, "\nmtllib color.mtl\n\n");

	fprintf(fpt, "g src\n");
	// save each vertex to .obj file
	for(i=0; i<NumVer1; i++)
		fprintf(fpt, "v %.6f %.6f %.6f\n", vertex1[i].coor[0], vertex1[i].coor[1], vertex1[i].coor[2]);

	fprintf(fpt, "\nusemtl red\n\n");
	// save each triangle (face)
	for(i=0; i<NumTri1; i++)
	{
		fprintf(fpt, "f");
		for(j=0; j<triangle1[i].NodeName; j++)	// record the number of vertex in this triangle
			fprintf(fpt, " %d", triangle1[i].v[j]+1);
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\n# dest model: %d vertices; %d triangles\n", NumVer2, NumTri2);
	fprintf(fpt, "g dest\n");
	// save each vertex to .obj file
	for(i=0; i<NumVer2; i++)
		fprintf(fpt, "v %.6f %.6f %.6f\n", vertex2[i].coor[0], vertex2[i].coor[1], vertex2[i].coor[2]);

	fprintf(fpt, "\nusemtl bule\n\n");
	// save each triangle (face)
	for(i=0; i<NumTri2; i++)
	{
		fprintf(fpt, "f");
		for(j=0; j<triangle2[i].NodeName; j++)	// record the number of vertex in this triangle
			fprintf(fpt, " %d", triangle2[i].v[j]+1+NumVer1);
		fprintf(fpt, "\n");
	}

	fclose(fpt);
}

