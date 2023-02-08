/*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

/*
 * rexec.c -- execute the fft
 */

/* $Id: rexec.c,v 1.26 1999/10/26 21:41:35 stevenj Exp $ */
#include <stdio.h>
#include <stdlib.h>

#include "fftw-int.h"
#include "rfftw.h"

void rfftw_strided_copy(int n, fftw_real *in, int ostride,
			fftw_real *out)
{
     int i;
     fftw_real r0, r1, r2, r3;

     i = 0;
     for (; i < (n & 3); ++i) {
	  out[i * ostride] = in[i];
     }
     for (; i < n; i += 4) {
	  r0 = in[i];
	  r1 = in[i + 1];
	  r2 = in[i + 2];
	  r3 = in[i + 3];
	  out[i * ostride] = r0;
	  out[(i + 1) * ostride] = r1;
	  out[(i + 2) * ostride] = r2;
	  out[(i + 3) * ostride] = r3;
     }
}

static void rexecutor_many(int n, fftw_real *in,
			   fftw_real *out,
			   fftw_plan_node *p,
			   int istride,
			   int ostride,
			   int howmany, int idist, int odist,
			   fftw_recurse_kind recurse_kind)
{
     int s;

     switch (p->type) {
	 case FFTW_REAL2HC:
	      {
		   fftw_real2hc_codelet *codelet = p->nodeu.real2hc.codelet;

		   HACK_ALIGN_STACK_ODD;
		   for (s = 0; s < howmany; ++s)
			codelet(in + s * idist, out + s * odist,
				out + n * ostride + s * odist,
				istride, ostride, -ostride);
		   break;
	      }

	 case FFTW_HC2REAL:
	      {
		   fftw_hc2real_codelet *codelet = p->nodeu.hc2real.codelet;

		   HACK_ALIGN_STACK_ODD;
		   for (s = 0; s < howmany; ++s)
			codelet(in + s * idist, in + n * istride + s * idist,
				out + s * odist,
				istride, -istride, ostride);
		   break;
	      }

	 default:
	      for (s = 0; s < howmany; ++s)
		   rfftw_executor_simple(n, in + s * idist,
					 out + s * odist,
					 p, istride, ostride,
					 recurse_kind);
     }
}

#ifdef FFTW_ENABLE_VECTOR_RECURSE

/* rexecutor_many_vector is like rexecutor_many, but it pushes the
   howmany loop down to the leaves of the transform: */
static void rexecutor_many_vector(int n, fftw_real *in,
				  fftw_real *out,
				  fftw_plan_node *p,
				  int istride,
				  int ostride,
				  int howmany, int idist, int odist)
{
     switch (p->type) {
	 case FFTW_REAL2HC:
	      {
		   fftw_real2hc_codelet *codelet = p->nodeu.real2hc.codelet;
		   int s;

		   HACK_ALIGN_STACK_ODD;
		   for (s = 0; s < howmany; ++s)
			codelet(in + s * idist, out + s * odist,
				out + n * ostride + s * odist,
				istride, ostride, -ostride);
		   break;
	      }

	 case FFTW_HC2REAL:
	      {
		   fftw_hc2real_codelet *codelet = p->nodeu.hc2real.codelet;
		   int s;

		   HACK_ALIGN_STACK_ODD;
		   for (s = 0; s < howmany; ++s)
			codelet(in + s * idist, in + n * istride + s * idist,
				out + s * odist,
				istride, -istride, ostride);
		   break;
	      }

	 case FFTW_HC2HC:
	      {
		   int r = p->nodeu.hc2hc.size;
		   int m = n / r;
		   int i;
		   fftw_hc2hc_codelet *codelet;
		   fftw_complex *W;

		   switch (p->nodeu.hc2hc.dir) {
		       case FFTW_REAL_TO_COMPLEX:
			    for (i = 0; i < r; ++i)
				 rexecutor_many_vector(m, in + i * istride,
						       out + i * (m*ostride),
						       p->nodeu.hc2hc.recurse,
						       istride * r, ostride,
						       howmany, idist, odist);

			    W = p->nodeu.hc2hc.tw->twarray;
			    codelet = p->nodeu.hc2hc.codelet;
			    HACK_ALIGN_STACK_EVEN;
			    for (i = 0; i < howmany; ++i)
				 codelet(out + i * odist, 
					 W, m * ostride, m, ostride);
			    break;
		       case FFTW_COMPLEX_TO_REAL:
			    W = p->nodeu.hc2hc.tw->twarray;
			    codelet = p->nodeu.hc2hc.codelet;
			    HACK_ALIGN_STACK_EVEN;
			    for (i = 0; i < howmany; ++i)
				 codelet(in + i * idist,
					 W, m * istride, m, istride);

			    for (i = 0; i < r; ++i)
				 rexecutor_many_vector(m, in + i * (m*istride),
						       out + i * ostride,
						       p->nodeu.hc2hc.recurse,
						       istride, ostride * r,
						       howmany, idist, odist);
			    break;
		       default:
			    goto bug;
		   }

		   break;
	      }

	 case FFTW_RGENERIC:
	      {
		   int r = p->nodeu.rgeneric.size;
		   int m = n / r;
		   int i;
		   fftw_rgeneric_codelet *codelet = p->nodeu.rgeneric.codelet;
		   fftw_complex *W = p->nodeu.rgeneric.tw->twarray;

		   switch (p->nodeu.rgeneric.dir) {
		       case FFTW_REAL_TO_COMPLEX:
			    for (i = 0; i < r; ++i)
				 rexecutor_many_vector(m, in + i * istride,
						 out + i * (m * ostride),
					       p->nodeu.rgeneric.recurse,
						   istride * r, ostride,
						       howmany, idist, odist);

			    for (i = 0; i < howmany; ++i)
				 codelet(out + i * odist, W, m, r, n, ostride);
			    break;
		       case FFTW_COMPLEX_TO_REAL:
			    for (i = 0; i < howmany; ++i)
				 codelet(in + i * idist, W, m, r, n, istride);

			    for (i = 0; i < r; ++i)
				 rexecutor_many_vector(m, in + i * m * istride,
						       out + i * ostride,
					       p->nodeu.rgeneric.recurse,
						   istride, ostride * r,
						       howmany, idist, odist);
			    break;
		       default:
			    goto bug;
		   }

		   break;
	      }

	 default:
	    bug:
	      fftw_die("BUG in rexecutor: invalid plan\n");
	      break;
     }
}

#endif /* FFTW_ENABLE_VECTOR_RECURSE */

void rfftw_executor_simple(int n, fftw_real *in,
			   fftw_real *out,
			   fftw_plan_node *p,
			   int istride,
			   int ostride,
			   fftw_recurse_kind recurse_kind)
{
     switch (p->type) {
	 case FFTW_REAL2HC:
	      HACK_ALIGN_STACK_ODD;
	      (p->nodeu.real2hc.codelet) (in, out, out + n * ostride,
					  istride, ostride, -ostride);
	      break;

	 case FFTW_HC2REAL:
	      HACK_ALIGN_STACK_ODD;
	      (p->nodeu.hc2real.codelet) (in, in + n * istride, out,
					  istride, -istride, ostride);
	      break;

	 case FFTW_HC2HC:
	      {
		   int r = p->nodeu.hc2hc.size;
		   int m = n / r;
		   /* 
		    * please do resist the temptation of initializing
		    * these variables here.  Doing so forces the
		    * compiler to keep a live variable across the
		    * recursive call.
		    */
		   fftw_hc2hc_codelet *codelet;
		   fftw_complex *W;

		   switch (p->nodeu.hc2hc.dir) {
		       case FFTW_REAL_TO_COMPLEX:
#ifdef FFTW_ENABLE_VECTOR_RECURSE
			    if (recurse_kind == FFTW_NORMAL_RECURSE)
#endif
				 rexecutor_many(m, in, out,
						p->nodeu.hc2hc.recurse,
						istride * r, ostride,
						r, istride, m * ostride,
						FFTW_NORMAL_RECURSE);
#ifdef FFTW_ENABLE_VECTOR_RECURSE
			    else
				 rexecutor_many_vector(m, in, out,
						p->nodeu.hc2hc.recurse,
						istride * r, ostride,
						r, istride, m * ostride);
#endif

			    W = p->nodeu.hc2hc.tw->twarray;
			    codelet = p->nodeu.hc2hc.codelet;
			    HACK_ALIGN_STACK_EVEN;
			    codelet(out, W, m * ostride, m, ostride);
			    break;
		       case FFTW_COMPLEX_TO_REAL:
			    W = p->nodeu.hc2hc.tw->twarray;
			    codelet = p->nodeu.hc2hc.codelet;
			    HACK_ALIGN_STACK_EVEN;
			    codelet(in, W, m * istride, m, istride);

#ifdef FFTW_ENABLE_VECTOR_RECURSE
			    if (recurse_kind == FFTW_NORMAL_RECURSE)
#endif
				 rexecutor_many(m, in, out,
						p->nodeu.hc2hc.recurse,
						istride, ostride * r,
						r, m * istride, ostride,
						FFTW_NORMAL_RECURSE);
#ifdef FFTW_ENABLE_VECTOR_RECURSE
			    else
				 rexecutor_many_vector(m, in, out,
						p->nodeu.hc2hc.recurse,
						istride, ostride * r,
						r, m * istride, ostride);
#endif
			    break;
		       default:
			    goto bug;
		   }

		   break;
	      }

	 case FFTW_RGENERIC:
	      {
		   int r = p->nodeu.rgeneric.size;
		   int m = n / r;
		   fftw_rgeneric_codelet *codelet = p->nodeu.rgeneric.codelet;
		   fftw_complex *W = p->nodeu.rgeneric.tw->twarray;

		   switch (p->nodeu.rgeneric.dir) {
		       case FFTW_REAL_TO_COMPLEX:
#ifdef FFTW_ENABLE_VECTOR_RECURSE
			    if (recurse_kind == FFTW_NORMAL_RECURSE)
#endif
				 rexecutor_many(m, in, out,
						p->nodeu.rgeneric.recurse,
						istride * r, ostride,
						r, istride, m * ostride,
						FFTW_NORMAL_RECURSE);
#ifdef FFTW_ENABLE_VECTOR_RECURSE
			    else
				 rexecutor_many_vector(m, in, out,
						p->nodeu.rgeneric.recurse,
						istride * r, ostride,
						r, istride, m * ostride);
#endif

			    codelet(out, W, m, r, n, ostride);
			    break;
		       case FFTW_COMPLEX_TO_REAL:
			    codelet(in, W, m, r, n, istride);

#ifdef FFTW_ENABLE_VECTOR_RECURSE
			    if (recurse_kind == FFTW_NORMAL_RECURSE)
#endif
				 rexecutor_many(m, in, out,
						p->nodeu.rgeneric.recurse,
						istride, ostride * r,
						r, m * istride, ostride,
						FFTW_NORMAL_RECURSE);
#ifdef FFTW_ENABLE_VECTOR_RECURSE
			    else
				 rexecutor_many_vector(m, in, out,
						p->nodeu.rgeneric.recurse,
						istride, ostride * r,
						r, m * istride, ostride);
#endif
			    break;
		       default:
			    goto bug;
		   }

		   break;
	      }

	 default:
	    bug:
	      fftw_die("BUG in rexecutor: invalid plan\n");
	      break;
     }
}

static void rexecutor_simple_inplace(int n, fftw_real *in,
				     fftw_real *out,
				     fftw_plan_node *p,
				     int istride,
				     fftw_recurse_kind recurse_kind)
{
     switch (p->type) {
	 case FFTW_REAL2HC:
	      HACK_ALIGN_STACK_ODD;
	      (p->nodeu.real2hc.codelet) (in, in, in + n * istride,
					  istride, istride, -istride);
	      break;

	 case FFTW_HC2REAL:
	      HACK_ALIGN_STACK_ODD;
	      (p->nodeu.hc2real.codelet) (in, in + n * istride, in,
					  istride, -istride, istride);
	      break;

	 default:
	      {
		   fftw_real *tmp;

		   if (out)
			tmp = out;
		   else
			tmp = (fftw_real *) fftw_malloc(n * sizeof(fftw_real));

		   rfftw_executor_simple(n, in, tmp, p, istride, 1, 
					 recurse_kind);
		   rfftw_strided_copy(n, tmp, istride, in);

		   if (!out)
			fftw_free(tmp);
	      }
     }
}

static void rexecutor_many_inplace(int n, fftw_real *in,
				   fftw_real *out,
				   fftw_plan_node *p,
				   int istride,
				   int howmany, int idist,
				   fftw_recurse_kind recurse_kind)
{
     switch (p->type) {
	 case FFTW_REAL2HC:
	      {
		   fftw_real2hc_codelet *codelet = p->nodeu.real2hc.codelet;
		   int s;

		   HACK_ALIGN_STACK_ODD;
		   for (s = 0; s < howmany; ++s)
			codelet(in + s * idist, in + s * idist,
				in + n * istride + s * idist,
				istride, istride, -istride);

		   break;
	      }

	 case FFTW_HC2REAL:
	      {
		   fftw_hc2real_codelet *codelet = p->nodeu.hc2real.codelet;
		   int s;

		   HACK_ALIGN_STACK_ODD;
		   for (s = 0; s < howmany; ++s)
			codelet(in + s * idist, in + n * istride + s * idist,
				in + s * idist,
				istride, -istride, istride);

		   break;
	      }

	 default:
	      {
		   int s;
		   fftw_real *tmp;
		   if (out)
			tmp = out;
		   else
			tmp = (fftw_real *) fftw_malloc(n * sizeof(fftw_real));

		   for (s = 0; s < howmany; ++s) {
			rfftw_executor_simple(n,
					      in + s * idist,
					      tmp,
					      p, istride, 1, recurse_kind);
			rfftw_strided_copy(n, tmp, istride, in + s * idist);
		   }

		   if (!out)
			fftw_free(tmp);
	      }
     }
}

/* user interface */
void rfftw(fftw_plan plan, int howmany, fftw_real *in, int istride,
	   int idist, fftw_real *out, int ostride, int odist)
{
     int n = plan->n;

     if (plan->flags & FFTW_IN_PLACE) {
	  if (howmany == 1) {
	       rexecutor_simple_inplace(n, in, out, plan->root, istride,
					plan->recurse_kind);
	  } else {
	       rexecutor_many_inplace(n, in, out, plan->root, istride, howmany,
				      idist, plan->recurse_kind);
	  }
     } else {
	  if (howmany == 1) {
	       rfftw_executor_simple(n, in, out, plan->root, istride, ostride,
				     plan->recurse_kind);
	  } else {
#ifdef FFTW_ENABLE_VECTOR_RECURSE
               int vector_size = plan->vector_size;
               if (vector_size <= 1)
#endif
		    rexecutor_many(n, in, out, plan->root, istride, ostride,
				   howmany, idist, odist,
				   plan->recurse_kind);
#ifdef FFTW_ENABLE_VECTOR_RECURSE
               else {
                    int s;
                    int num_vects = howmany / vector_size;
                    fftw_plan_node *root = plan->root;

                    for (s = 0; s < num_vects; ++s)
                         rexecutor_many_vector(n,
					       in + s * (vector_size * idist),
					       out + s * (vector_size * odist),
					       root,
					       istride, ostride,
					       vector_size, idist, odist);

                    s = howmany % vector_size;
                    if (s > 0)
                         rexecutor_many(n,
					in + num_vects * (vector_size*idist),
					out + num_vects * (vector_size*odist),
					root,
					istride, ostride,
					s, idist, odist,
					FFTW_NORMAL_RECURSE);
               }
#endif
	  }
     }
}

void rfftw_one(fftw_plan plan, fftw_real *in, fftw_real *out)
{
     int n = plan->n;

     if (plan->flags & FFTW_IN_PLACE)
	  rexecutor_simple_inplace(n, in, out, plan->root, 1,
				   plan->recurse_kind);
     else
	  rfftw_executor_simple(n, in, out, plan->root, 1, 1,
				plan->recurse_kind);
}
