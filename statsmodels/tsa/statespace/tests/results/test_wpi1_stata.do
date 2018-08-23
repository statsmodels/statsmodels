use https://www.stata-press.com/data/r12/wpi1, clear
gen dwpi = D.wpi

// Estimate an AR(3) via a state-space model
// (to test prediction, standardized residuals, predicted states)
constraint 1 [dwpi]u1 = 1
constraint 2 [u2]L.u1 = 1
constraint 3 [u3]L.u2 = 1

sspace (u1 L.u1 L.u2 L.u3, state noconstant) ///
       (u2 L.u1, state noconstant noerror) ///
       (u3 L.u2, state noconstant noerror) ///
       (dwpi u1, noconstant noerror), ///
       constraints(1/3) covstate(diagonal)

predict dep*
predict sr*, rstandard
predict sp*, states smethod(onestep) // predicted states
predict sf*, states smethod(filter)  // filtered states
predict sm*, states smethod(smooth)  // smoothed states

outsheet using results_wpi1_ar3_stata.csv, comma
