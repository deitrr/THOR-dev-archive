// ==============================================================================
// This file is part of THOR.
//
//     THOR is free software : you can redistribute it and / or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.
//
//     THOR is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//     GNU General Public License for more details.
//
//     You find a copy of the GNU General Public License in the main
//     THOR directory under <license.txt>.If not, see
//     <http://www.gnu.org/licenses/>.
// ==============================================================================
//
// Defines Planet's properties
//
//
// Description: Planet parameters.
//
// Method: -
//
//
// Known limitations: None
//
//
// Known issues: None
//
//
// If you use this code please cite the following reference: 
//
//       [1] Mendonca, J.M., Grimm, S.L., Grosheintz, L., & Heng, K., ApJ, 829, 115, 2016  
//
// Current Code Owner: Joao Mendonca, EEG. joao.mendonca@csh.unibe.ch
//
// History:
// Version Date       Comment
// ======= ====       =======
//
// 1.0     16/08/2017 Released version  (JM)
//
////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>

#include "../headers/planet.h"

XPlanet::XPlanet(){
//
//  Earth
    if(planetnumber==1){
        // ID
        sprintf(simulation_ID, "%s", "Earth");
        //////////////
        // BULK     //
        //////////////
        A = 6371000.0    ; // Radius [m]
        Omega = 7.292E-5 ; // Rotation rate [s-1]
        Gravit= 9.8      ; // Gravitational acceleration [m/s^2]
        ////////////////
        // ATMOSPHERE //
        ////////////////
        Mmol = 28.964         ; // Mean molecular mass of dry air [kg]
        Rd   = 287.0          ; // Gas constant [J/(Kg K)]
        Cp   = 1005.0         ; // Specific heat capacities [J/(Kg K)]
        Tmean= 300.0          ; // Mean atmospheric temperature [K]
        P_Ref = 100000.0      ; // Reference surface pressure [Pa] 
        Top_altitude = 36000.0; // Altitude of the top of the model domain [m]
        Diffc = 0.007         ; // Strength of diffusion
    }    
}