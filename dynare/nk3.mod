// dynare/nk3.mod — Three-equation New Keynesian model (Galí 2015, ch. 3)

var y_gap infl i_nom r_real v;     // 5 endogenous variables -> 5 equations
varexo eps_v;                      // monetary policy shock

parameters betta siggma varphi alppha epsilon theta phi_pi phi_y rho_v 
Theta lambda kappa;

// Calibration (Galí 2015, ch. 3), quarterly
betta = 0.99; siggma = 1; varphi = 5; alppha = 0.25; epsilon = 9; 
theta = 0.75; phi_pi = 1.5; phi_y = 0.125; rho_v = 0.5;

// phi_pi < 1 violates the Taylor principle: Blanchard-Kahn fails (indeterminacy)

// Derived parameters: like a Python constant computed from others, 
//order matters
Theta  = (1 - alppha) / (1 - alppha + alppha*epsilon);
lambda = (1 - theta) * (1 - betta*theta) / theta * Theta;
kappa  = lambda * (siggma + (varphi + alppha) / (1 - alppha));

model(linear);
    // New Keynesian Phillips curve
    infl = betta*infl(+1) + kappa*y_gap;
    // Dynamic IS curve
    y_gap = y_gap(+1) - (1/siggma)*(i_nom - infl(+1));
    // Taylor rule
    i_nom = phi_pi*infl + phi_y*y_gap + v;
    // Ex-ante real interest rate
    r_real = i_nom - infl(+1);
    // Monetary policy shock, AR(1)
    v = rho_v*v(-1) + eps_v;
end;

check;

shocks;
    var eps_v; stderr 0.25;         // 25 bp per quarter = 1 pp annualised, as in Galí
end;

stoch_simul(order=1, irf=16) y_gap infl i_nom r_real;