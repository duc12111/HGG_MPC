/*
 * AD tool to FORCESPRO Template - missing information to be filled in by createADTool.m 
 * (C) embotech AG, Zurich, Switzerland, 2013-2022. All rights reserved.
 *
 * This file is part of the FORCESPRO client, and carries the same license.
 */ 

#ifdef __cplusplus
extern "C" {
#endif
    
#include "include/FORCESNLPsolver3.h"

#ifndef NULL
#define NULL ((void *) 0)
#endif

#include "FORCESNLPsolver3_model.h"



/* copies data from sparse matrix into a dense one */
static void FORCESNLPsolver3_sparse2fullcopy(solver_int32_default nrow, solver_int32_default ncol, const solver_int32_default *colidx, const solver_int32_default *row, FORCESNLPsolver3_callback_float *data, FORCESNLPsolver3_float *out)
{
    solver_int32_default i, j;
    
    /* copy data into dense matrix */
    for(i=0; i<ncol; i++)
    {
        for(j=colidx[i]; j<colidx[i+1]; j++)
        {
            out[i*nrow + row[j]] = ((FORCESNLPsolver3_float) data[j]);
        }
    }
}




/* AD tool to FORCESPRO interface */
extern solver_int32_default FORCESNLPsolver3_adtool2forces(FORCESNLPsolver3_float *x,        /* primal vars                                         */
                                 FORCESNLPsolver3_float *y,        /* eq. constraint multiplers                           */
                                 FORCESNLPsolver3_float *l,        /* ineq. constraint multipliers                        */
                                 FORCESNLPsolver3_float *p,        /* parameters                                          */
                                 FORCESNLPsolver3_float *f,        /* objective function (scalar)                         */
                                 FORCESNLPsolver3_float *nabla_f,  /* gradient of objective function                      */
                                 FORCESNLPsolver3_float *c,        /* dynamics                                            */
                                 FORCESNLPsolver3_float *nabla_c,  /* Jacobian of the dynamics (column major)             */
                                 FORCESNLPsolver3_float *h,        /* inequality constraints                              */
                                 FORCESNLPsolver3_float *nabla_h,  /* Jacobian of inequality constraints (column major)   */
                                 FORCESNLPsolver3_float *hess,     /* Hessian (column major)                              */
                                 solver_int32_default stage,     /* stage number (0 indexed)                           */
								 solver_int32_default iteration, /* iteration number of solver                         */
								 solver_int32_default threadID   /* Id of caller thread                                */)
{
    /* AD tool input and output arrays */
    const FORCESNLPsolver3_callback_float *in[4];
    FORCESNLPsolver3_callback_float *out[7];
	

	/* Allocate working arrays for AD tool */
	FORCESNLPsolver3_float w[40];
	
    /* temporary storage for AD tool sparse output */
    FORCESNLPsolver3_callback_float this_f;
    FORCESNLPsolver3_float nabla_f_sparse[10];
    FORCESNLPsolver3_float h_sparse[3];
    FORCESNLPsolver3_float nabla_h_sparse[12];
    FORCESNLPsolver3_float c_sparse[1];
    FORCESNLPsolver3_float nabla_c_sparse[1];
            
    
    /* pointers to row and column info for 
     * column compressed format used by AD tool */
    solver_int32_default nrow, ncol;
    const solver_int32_default *colind, *row;
    
    /* set inputs for AD tool */
    in[0] = x;
    in[1] = p;
    in[2] = l;
    in[3] = y;

	if ((0 <= stage && stage <= 6))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		FORCESNLPsolver3_objective_0(in, out, NULL, w, 0);
		if( nabla_f )
		{
			nrow = FORCESNLPsolver3_objective_0_sparsity_out(1)[0];
			ncol = FORCESNLPsolver3_objective_0_sparsity_out(1)[1];
			colind = FORCESNLPsolver3_objective_0_sparsity_out(1) + 2;
			row = FORCESNLPsolver3_objective_0_sparsity_out(1) + 2 + (ncol + 1);
			FORCESNLPsolver3_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		FORCESNLPsolver3_rkfour_0(x, p, c, nabla_c, FORCESNLPsolver3_cdyn_0rd_0, FORCESNLPsolver3_cdyn_0, threadID);
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		FORCESNLPsolver3_inequalities_0(in, out, NULL, w, 0);
		if( h )
		{
			nrow = FORCESNLPsolver3_inequalities_0_sparsity_out(0)[0];
			ncol = FORCESNLPsolver3_inequalities_0_sparsity_out(0)[1];
			colind = FORCESNLPsolver3_inequalities_0_sparsity_out(0) + 2;
			row = FORCESNLPsolver3_inequalities_0_sparsity_out(0) + 2 + (ncol + 1);
			FORCESNLPsolver3_sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h )
		{
			nrow = FORCESNLPsolver3_inequalities_0_sparsity_out(1)[0];
			ncol = FORCESNLPsolver3_inequalities_0_sparsity_out(1)[1];
			colind = FORCESNLPsolver3_inequalities_0_sparsity_out(1) + 2;
			row = FORCESNLPsolver3_inequalities_0_sparsity_out(1) + 2 + (ncol + 1);
			FORCESNLPsolver3_sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}
	if ((7 == stage))
	{
		
		
		out[0] = &this_f;
		out[1] = nabla_f_sparse;
		FORCESNLPsolver3_objective_1(in, out, NULL, w, 0);
		if( nabla_f )
		{
			nrow = FORCESNLPsolver3_objective_1_sparsity_out(1)[0];
			ncol = FORCESNLPsolver3_objective_1_sparsity_out(1)[1];
			colind = FORCESNLPsolver3_objective_1_sparsity_out(1) + 2;
			row = FORCESNLPsolver3_objective_1_sparsity_out(1) + 2 + (ncol + 1);
			FORCESNLPsolver3_sparse2fullcopy(nrow, ncol, colind, row, nabla_f_sparse, nabla_f);
		}
		
		out[0] = h_sparse;
		out[1] = nabla_h_sparse;
		FORCESNLPsolver3_inequalities_1(in, out, NULL, w, 0);
		if( h )
		{
			nrow = FORCESNLPsolver3_inequalities_1_sparsity_out(0)[0];
			ncol = FORCESNLPsolver3_inequalities_1_sparsity_out(0)[1];
			colind = FORCESNLPsolver3_inequalities_1_sparsity_out(0) + 2;
			row = FORCESNLPsolver3_inequalities_1_sparsity_out(0) + 2 + (ncol + 1);
			FORCESNLPsolver3_sparse2fullcopy(nrow, ncol, colind, row, h_sparse, h);
		}
		if( nabla_h )
		{
			nrow = FORCESNLPsolver3_inequalities_1_sparsity_out(1)[0];
			ncol = FORCESNLPsolver3_inequalities_1_sparsity_out(1)[1];
			colind = FORCESNLPsolver3_inequalities_1_sparsity_out(1) + 2;
			row = FORCESNLPsolver3_inequalities_1_sparsity_out(1) + 2 + (ncol + 1);
			FORCESNLPsolver3_sparse2fullcopy(nrow, ncol, colind, row, nabla_h_sparse, nabla_h);
		}
	}
    
    /* add to objective */
    if (f != NULL)
    {
        *f += ((FORCESNLPsolver3_float) this_f);
    }

    return 0;
}

#ifdef __cplusplus
} /* extern "C" */
#endif
