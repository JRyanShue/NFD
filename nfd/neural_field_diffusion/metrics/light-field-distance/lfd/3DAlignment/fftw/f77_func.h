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

#ifndef F77_FUNC_H
#define F77_FUNC_H

#include "fftw-int.h"

/* Define a macro to mangle function names so that they can be
   recognized by the Fortran linker.  Specifically, F77_FUNC_
   is designed to mangle identifiers containing an underscore. */

#ifdef FFTW_FORTRANIZE_LOWERCASE
#  ifdef FFTW_FORTRANIZE_EXTRA_UNDERSCORE
#    define F77_FUNC_(x,X) x ## _
#  else
#    define F77_FUNC_(x,X) x
#  endif
#endif

#ifdef FFTW_FORTRANIZE_LOWERCASE_UNDERSCORE
#  ifdef FFTW_FORTRANIZE_EXTRA_UNDERSCORE
#    define F77_FUNC_(x,X) x ## __
#  else
#    define F77_FUNC_(x,X) x ## _
#  endif
#endif

#ifdef FFTW_FORTRANIZE_UPPERCASE
#  ifdef FFTW_FORTRANIZE_EXTRA_UNDERSCORE
#    define F77_FUNC_(x,X) X ## _
#  else
#    define F77_FUNC_(x,X) X
#  endif
#endif

#ifdef FFTW_FORTRANIZE_UPPERCASE_UNDERSCORE
#  ifdef FFTW_FORTRANIZE_EXTRA_UNDERSCORE
#    define F77_FUNC_(x,X) X ## __
#  else
#    define F77_FUNC_(x,X) X ## _
#  endif
#endif

#endif /* F77_FUNC_H */
