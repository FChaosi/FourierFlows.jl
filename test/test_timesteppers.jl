function constantdiffusionproblem(stepper; nx=128, Lx=2π, kappa=1e-2, nsteps=1000, dev=CPU())
   τ = 1/kappa  # time-scale for diffusive decay
  dt = 1e-9 * τ # dynamics are resolved

  prob = Diffusion.Problem(nx=nx, Lx=Lx, kappa=kappa, dt=dt, stepper=stepper, dev=dev)
  g = prob.grid

  # a gaussian initial condition c(x, t=0)
  c0ampl, σ = 0.01, 0.2
  c0func(x) = @. c0ampl*exp(-x^2/(2σ^2))
  c0 = c0func.(g.x)

  # analytic solution for for 1D heat equation with constant κ
  tfinal = nsteps*dt
  σt = sqrt(2*kappa*tfinal + σ^2)
  cfinal = @. c0ampl*σ/σt * exp(-g.x^2/(2*σt^2))

  Diffusion.set_c!(prob, c0)
  tcomp = @elapsed stepforward!(prob, nsteps)
  Diffusion.updatevars!(prob)

  prob, c0, cfinal, nsteps, tcomp
end

function constantdiffusionproblem2D(stepper; nx=128, ny=64, Lx=2π, Ly=4π, kappa=1e-2, nsteps=1000, dev=CPU())
    τ = 1/kappa  # time-scale for diffusive decay
    dt = 1e-9 * τ # dynamics are resolved

prob = Diffusion2D.Problem(nx=nx, ny=ny, Lx=Lx, Ly=Ly, kappa=kappa, dt=dt, stepper=stepper, dev=dev)
g = prob.grid

#Gaussian Initial Condition
c0, σ0 = 0.1, 0.2
c1, σ1 = 0.1, 0.2

c0_x_func(x) = @. c0*exp(-x^2/(2*(σ0)^2))
c0_y_func(y) = @. c1*exp(-y^2/(2*(σ1)^2))

c0_val = c0_x_func.(g.x) .* c0_y_func.(g.y)

#Analytic Solution
tfinal = nsteps*dt

σ0t = sqrt(2*kappa*tfinal + (σ0)^2)
σ1t = sqrt(2*kappa*tfinal + (σ1)^2)

cfinal_x = @. c0 * (σ0/σ0t) * exp(-g.x^2/(2*(σ0t)^2))
cfinal_y = @. c1 * (σ1/σ1t) * exp(-g.y^2/(2*(σ1t)^2))

cfinal_val = cfinal_x .* cfinal_y

#Numerical Solution
Diffusion2D.set_c!(prob, c0_val)
tcomp = @elapsed stepforward!(prob, nsteps)
Diffusion2D.updatevars!(prob)

prob, c0_val, cfinal_val, nsteps, tcomp
end 

function varyingdiffusionproblem(stepper; nx=128, Lx=2π, kappa=1e-2, nsteps=1000, dev=CPU())
   τ = 1/kappa  # time-scale for diffusive decay
  dt = 1e-9 * τ # dynamics are resolved

  kappa = kappa*ones(nx) # this is actually a constant diffusion but defining it
                         # as an array makes stepforward! call function calcN!
                         # instead of just the linear coefficients L*sol

  prob = Diffusion.Problem(nx=nx, Lx=Lx, kappa=kappa, dt=dt, stepper=stepper, dev=dev)
  g = prob.grid

  # a gaussian initial condition c(x, t=0)
  c0ampl, σ = 0.01, 0.2
  c0func(x) = @. c0ampl*exp(-x^2/(2σ^2))
  c0 = c0func.(g.x)

  # analytic solution for for 1D heat equation with constant κ
  tfinal = nsteps*dt
  σt = sqrt(2*kappa[1]*tfinal + σ^2)
  cfinal = @. c0ampl*σ/σt * exp(-g.x^2/(2*σt^2))

  Diffusion.set_c!(prob, c0)
  tcomp = @elapsed stepforward!(prob, nsteps)
  Diffusion.updatevars!(prob)

  prob, c0, cfinal, nsteps, tcomp
end


function constantdiffusiontest(stepper, dev::Device=CPU(); kwargs...)
  prob, c0, c1, nsteps, tcomp = constantdiffusionproblem(stepper; kwargs...)
  normmsg = "$stepper: relative error ="
  @printf("% 40s %.2e (%.3f s)\n", normmsg, norm(c1-prob.vars.c)/norm(c1), tcomp)
  isapprox(c1, prob.vars.c, rtol=nsteps*rtol_timesteppers)
end

function constantdiffusiontest2D(stepper, dev::Device=CPU(); kwargs...)
    prob, ci, cf, nsteps, tcomp = constantdiffusionproblem2D(stepper; kwargs...)
    normmsg = "$stepper: relative error ="
    @printf("% 40s %.2e (%.3f s)\n", normmsg, norm(cf-prob.vars.c)/norm(cf), tcomp)
    isapprox(cf, prob.vars.c, rtol=nsteps*rtol_timesteppers)
end

function varyingdiffusiontest(stepper, dev::Device=CPU(); kwargs...)
  prob, c0, c1, nsteps, tcomp = varyingdiffusionproblem(stepper; kwargs...)
  normmsg = "$stepper: relative error ="
  @printf("% 40s %.2e (%.3f s)\n", normmsg, norm(c1-prob.vars.c)/norm(c1), tcomp)
  isapprox(c1, prob.vars.c, rtol=nsteps*rtol_timesteppers)
end
