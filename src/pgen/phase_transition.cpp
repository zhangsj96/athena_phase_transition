//======================================================================================
/* Athena++ astrophysical MHD code
 * Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
 *
 * This program is free software: you can redistribute and/or modify it under the terms
 * of the GNU General Public License (GPL) as published by the Free Software Foundation,
 * either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 * PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 * You should have received a copy of GNU GPL in the file LICENSE included in the code
 * distribution.  If not see <http://www.gnu.org/licenses/>.
 *====================================================================================*/

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>      // sqrt
#include <fstream>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/hydro_srcterms.hpp"
#include "../mesh/mesh.hpp"
#include "../nr_radiation/integrators/rad_integrators.hpp"
#include "../nr_radiation/radiation.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"


//========================================================================================
Real L_T = 3e10; // erg/g
// m_H = 1.6735575e-24
// k_b = 1.38064852e-16
// mu  = 30
Real kb_over_mu_mH = 2749927.464896386;
Real C_v = 3.*kb_over_mu_mH;
const int max_iter = 50;
Real tol = 1e-6;
Real tunit; // temperature unit in K
Real rhounit; // density unit in g/cm^3
Real rho0_gas;
Real rho0_par;

// Cooling related constants
Real sigma_b = 5.6704e-5; // erg/cm^2/s/K^4
Real kappa_par; // opacity
Real lunit; // length unit in cm
Real amplitude; // amplitude of perturbation
Real wavelength; // wavelength of perturbation


//========================================================================================
// \brief help functions
Real lnPsat(Real T);
Real Psat(Real T);
Real rho_sat(Real T);
Real dlnPsat_dT(Real T);
Real f_energy(Real T, Real rho_g, Real rho_tot, Real T_gas, const Real dt);
Real dfdT(Real T, Real rho_tot, const Real dt);
Real newton_solver(Real T_guess, Real rho_g, Real rho_tot, Real T_gas, int max_iter, Real tol, const Real dt);

// Phase transition source function
void PhaseChange(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

// Optical thin cooling function
void Cooling(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

// Source functions combined
void SourceFunction(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar);

//======================================================================================
//! \fn Real lnPsat(Real T)
//  \brief the input T is in cgs (K) pressure in Ba
Real lnPsat(Real T) {
  return -56.82 - 18946/T + 9.3974*std::log(T);
}

Real Psat(Real T) {
  return std::exp(lnPsat(T));
}

Real rho_sat(Real T) {
  return 1./kb_over_mu_mH * Psat(T)/T;
}

Real dlnPsat_dT(Real T) {
  return 18946./SQR(T) + 9.3974/T;
}

Real f_energy(Real T, Real rho_g, Real rho_tot, Real T_gas, const Real dt) {
  return rho_sat(T) - rho_g + C_v/L_T*rho_tot*(T-T_gas) 
         + 4.*sigma_b/L_T*std::pow(T,4.0)*(rho_tot-rho_g)*kappa_par*dt;
}

Real dfdT(Real T, Real rho_tot, const Real dt) {
  Real vunit = std::sqrt(kb_over_mu_mH * tunit);
  Real timeunit = lunit/vunit;
  return (dlnPsat_dT(T) - 1./T)*rho_sat(T) + C_v/L_T*rho_tot
         + 4.*sigma_b*kappa_par*timeunit*dt/L_T 
         * (-(dlnPsat_dT(T) - 1./T)*rho_sat(T)*std::pow(T,4.0) + 4.*(rho_tot-rho_sat(T)*std::pow(T,3.0)));
}

Real newton_solver(Real T_guess, Real rho_g, Real rho_tot, Real T_gas, int max_iter, Real tol, const Real dt) {
  for (int i = 0; i < max_iter; ++i) {
    Real fval = f_energy(T_guess, rho_g, rho_tot, T_gas, dt);
    Real dfdT_val = dfdT(T_guess, rho_tot, dt);
    Real delta = -fval / dfdT_val;
    T_guess += delta;
    if (std::abs(delta) < tol) {
      std::cout << "Converged in " << i + 1 << " steps.\n";
      return T_guess;
    }
  }
  std::cerr << "Newton method did not converge.\n";
  return T_guess;
}

//======================================================================================
// \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief
//=======================================================================================
void Mesh::InitUserMeshData(ParameterInput *pin) {
  rho0_gas = pin->GetOrAddReal("problem","rho0_gas", 1.0);
  rho0_par = pin->GetOrAddReal("problem","rho0_par", 1.0);
  tunit = pin->GetOrAddReal("problem","tunit", 2300.0);
  lunit = pin->GetOrAddReal("problem","lunit", 1.0e5);
  wavelength = pin->GetOrAddReal("problem","wavelength", 0.5);
  amplitude = pin->GetOrAddReal("problem","amplitude", 1e-6);
  kappa_par = pin->GetOrAddReal("problem","kappa_par", 2.5);
  Real tgas;
  tgas = pin->GetOrAddReal("problem","tgas",1.0);
  rhounit = rho_sat(tgas*tunit);
  EnrollUserExplicitSourceFunction(SourceFunction);
  return;
}

//======================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief
//======================================================================================
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Real tgas, er, sigma;

  er = pin->GetOrAddReal("problem","er",10.0);
  tgas = pin->GetOrAddReal("problem","tgas",1.0);
  sigma = pin->GetOrAddReal("problem","sigma",100.0);
  Real gamma = peos->GetGamma();

  Real vunit = std::sqrt(kb_over_mu_mH * tunit);
  Real timeunit = lunit/vunit;
  // some hard coded constants
  Real a = 16.6;
  Real b = 0.3;
  Real l = 1.6;
  Real big_Gamma = (1.0+a)*(b+l)/(1.0+l*a);
  Real omegaT = 4.*sigma_b*std::pow(tgas*tunit, 3.0) * kappa_par/C_v * timeunit;
  Real omegaL = a*omegaT/(1.0+l*a);
  Real omegaIm= omegaL*(1.0+a)/(big_Gamma*a);
  Real wavenumber = 2.0*PI/wavelength;

  // Initialize hydro variable
  for(int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x1 = pcoord->x1v(i);
        phydro->u(IDN,k,j,i) = rho0_gas * (1.0 + a*amplitude*std::sin(2.0*PI*x1/wavelength));
        phydro->u(IM1,k,j,i) = 0.0 + (wavenumber/omegaIm)*(1.0+a)*amplitude*std::sin(2.0*PI*x1/wavelength-PI/2.0);
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        if (NON_BAROTROPIC_EOS) {
          phydro->u(IEN,k,j,i) = rho0_gas*tgas/(gamma-1.0) * (1.0+(1.0+a)*amplitude*std::sin(2.0*PI*x1/wavelength));
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM1,k,j,i))/phydro->u(IDN,k,j,i);
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM2,k,j,i))/phydro->u(IDN,k,j,i);
          phydro->u(IEN,k,j,i) += 0.5*SQR(phydro->u(IM3,k,j,i))/phydro->u(IDN,k,j,i);
        }
        
        if (NSCALARS > 0) {
          for (int n=0; n<NSCALARS; ++n) {
            pscalars->s(n,k,j,i) = rho0_par + rho0_gas*(-SQR(wavenumber/omegaIm)*(1.0+a)/a-1.0)*a*amplitude*std::sin(2.0*PI*x1/wavelength);
          }
        } 
      }
    }
  }
  // Now initialize opacity and specific intensity
  if (NR_RADIATION_ENABLED || IM_RADIATION_ENABLED) {
    int nfreq = pnrrad->nfreq;
    //int nang = pnrrad->nang;
    AthenaArray<Real> ir_cm;
    ir_cm.NewAthenaArray(pnrrad->n_fre_ang);
    Real *ir_lab;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=is; i<=ie; ++i) {
          ir_lab = &(pnrrad->ir(k,j,i,0));
          for (int n=0; n<pnrrad->n_fre_ang; n++) {
             ir_lab[n] = er;
          }
        }
      }
    }

    for (int k=0; k<ncells3; ++k) {
      for (int j=0; j<ncells2; ++j) {
        for (int i=0; i<ncells1; ++i) {
          for (int ifr=0; ifr < nfreq; ++ifr) {
            pnrrad->sigma_s(k,j,i,ifr) = 0.0;
            pnrrad->sigma_a(k,j,i,ifr) = sigma;
            pnrrad->sigma_pe(k,j,i,ifr) = sigma;
            pnrrad->sigma_p(k,j,i,ifr) = sigma;
          }
        }
      }
    }
    ir_cm.DeleteAthenaArray();
  }
  return;
}

//======================================================================================
// \fn void PhaseChange(MeshBlock *pmb, const Real time, const Real dt,
//                      const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
//                      const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
//                      AthenaArray<Real> &cons_scalar)
// \brief phase change function
void PhaseChange(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar){
  Real gamma_gas = pmb->peos->GetGamma();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real ekin = 0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                                          +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
        Real eint = cons(IEN,k,j,i)-ekin;
        Real T_gas = eint*(gamma_gas-1.0)/cons(IDN,k,j,i) * tunit; // cgs unit
        // std::cout << "Sum of gas and particle density before the iteration: " << cons(IDN,k,j,i)+cons_scalar(0,k,j,i) << std::endl;
        Real rho_g = cons(IDN,k,j,i) * rhounit; // cgs uni
        Real rho_p = cons_scalar(0,k,j,i) * rhounit; // cgs unit
        Real rho_tot = rho_g + rho_p;  // cgs unit
        Real T_guess = T_gas; // cgs unit
        Real T_new = newton_solver(T_guess, rho_g,
                                   rho_tot, T_gas, 
                                   max_iter, tol, dt); // cgs unit
        Real rho_g_new = rho_sat(T_new); // cgs unit
        Real rho_p_new = rho_tot - rho_g_new; // cgs unit
        if (rho_g_new > rho_tot) {
          rho_g_new = rho_tot;
          rho_p_new = 0.0;
          Real drho = rho_g_new - rho_g;
          Real dT = -drho/rho_tot * 1./(C_v/L_T);
          T_new = T_gas + dT;
        }
        if (rho_g_new < 0.0) {
          rho_g_new = 0.0;
          rho_p_new = rho_tot;
          Real drho = rho_g_new - rho_g;
          Real dT = -drho/rho_tot * 1./(C_v/L_T);
          T_new = T_gas + dT;
        }
        cons(IDN,k,j,i) = rho_g_new / rhounit;
        cons_scalar(0,k,j,i) = rho_p_new / rhounit;
        // std::cout << "Sum of gas and particle density after the iteration: " << cons(IDN,k,j,i)+cons_scalar(0,k,j,i) << std::endl;
        Real eint_new = cons(IDN,k,j,i) * T_new/tunit/(gamma_gas-1.0);
        cons(IEN,k,j,i) = eint_new + ekin;
      }
    }
  }
  return;    
}

// \fn void Cooling(MeshBlock *pmb, const Real time, const Real dt,
//                  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
//                  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
//                  AthenaArray<Real> &cons_scalar)
// \brief cooling function
void Cooling(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar){
  Real vunit = std::sqrt(kb_over_mu_mH * tunit);
  Real timeunit = lunit/vunit;
  Real gamma_gas = pmb->peos->GetGamma();
  for (int k = pmb->ks; k <= pmb->ke; ++k) {
    for (int j = pmb->js; j <= pmb->je; ++j) {
      for (int i = pmb->is; i <= pmb->ie; ++i) {
        Real ekin = 0.5*(SQR(cons(IM1,k,j,i))+SQR(cons(IM2,k,j,i))
                    +SQR(cons(IM3,k,j,i)))/cons(IDN,k,j,i);
        Real eint = cons(IEN,k,j,i)-ekin;
        Real T_gas = eint*(gamma_gas-1.0)/cons(IDN,k,j,i); // temperature code unit
        Real dE = -4.*sigma_b*std::pow(T_gas,4.0)*cons_scalar(0,k,j,i)*kappa_par * dt; //hybrid unit
        dE = dE * std::pow(tunit, 3.0) / kb_over_mu_mH * timeunit;
        eint += dE;
        if (eint < 0.0) {
          eint = 0.0;
        }
        cons(IEN,k,j,i) = eint + ekin;
      }
    }
  }
  return;
}

// \fn void SourceFunction(MeshBlock *pmb, const Real time, const Real dt,
//                  const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
//                  const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
//                  AthenaArray<Real> &cons_scalar)
// \brief combined source function
void SourceFunction(MeshBlock *pmb, const Real time, const Real dt,
              const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
              const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
              AthenaArray<Real> &cons_scalar){
  // Cooling(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  PhaseChange(pmb, time, dt, prim, prim_scalar, bcc, cons, cons_scalar);
  return;
}
