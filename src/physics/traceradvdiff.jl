module TracerAdvDiff

using FourierFlows
export Params, Vars, Equation, set_c!, updatevars!

abstract type AbstractTracerParams <: AbstractParams end

# Problems
function ConstDiffSteadyFlowProblem(;
  grid = nothing,
    nx = 128,
    Lx = 2π,
    ny = nx,
    Ly = Lx,
     κ = 1.0,
     η = κ,
     u = nothing,
     v = nothing,
    dt = 0.01,
  stepper = "RK4"
  )

  # Defaults
  if u == nothing; uin(x, y) = 0.0
  else;            uin = u
  end

  if v == nothing; vin(x, y) = 0.0
  else;            vin = v
  end

  if grid == nothing; g = TwoDGrid(nx, Lx, ny, Ly)
  else;               g = grid
  end

  vs = TracerAdvDiff.Vars(g)
  pr = TracerAdvDiff.ConstDiffSteadyFlowParams(η, κ, uin, vin, g)
  eq = TracerAdvDiff.Equation(pr, g)
  ts = FourierFlows.autoconstructtimestepper(stepper, dt, eq.LC, g)
  FourierFlows.Problem(g, vs, pr, eq, ts)
end


# Problems
function ConstDiffProblem(;
  grid = nothing,
    nx = 128,
    Lx = 2π,
    ny = nx,
    Ly = Lx,
     κ = 1.0,
     η = κ,
     u = nothing,
     v = nothing,
    dt = 0.01,
  stepper = "RK4"
  )

  # Defaults
  if u == nothing; uin(x, y) = 0.0
  else;            uin = u
  end

  if v == nothing; vin(x, y) = 0.0
  else;            vin = v
  end

  if grid == nothing; g = TwoDGrid(nx, Lx, ny, Ly)
  else;               g = grid
  end

  vs = TracerAdvDiff.Vars(g)
  pr = TracerAdvDiff.ConstDiffParams(η, κ, uin, vin)
  eq = TracerAdvDiff.Equation(pr, g)
  ts = FourierFlows.autoconstructtimestepper(stepper, dt, eq.LC, g)
  FourierFlows.Problem(g, vs, pr, eq, ts)
end


# Params
struct ConstDiffParams <: AbstractTracerParams
  η::Float64                   # Constant isotropic horizontal diffusivity
  κ::Float64                   # Constant isotropic vertical diffusivity
  κh::Float64                  # Constant isotropic hyperdiffusivity
  nκh::Float64                 # Constant isotropic hyperdiffusivity order
  u::Function                  # Advecting x-velocity
  v::Function                  # Advecting y-velocity
end
ConstDiffParams(η, κ, u, v) = ConstDiffParams(η, κ, 0, 0, u, v)


struct ConstDiffSteadyFlowParams <: AbstractTracerParams
  η::Float64                   # Constant horizontal diffusivity
  κ::Float64                   # Constant vertical diffusivity
  κh::Float64                  # Constant isotropic hyperdiffusivity
  nκh::Float64                 # Constant isotropic hyperdiffusivity order
  u::Array{Float64,2}          # Advecting x-velocity
  v::Array{Float64,2}          # Advecting y-velocity
end

function ConstDiffSteadyFlowParams(η, κ, κh, nκh, u::Function, v::Function, g::TwoDGrid)
  ugrid = u.(g.X, g.Y)
  vgrid = v.(g.X, g.Y)
  ConstDiffSteadyFlowParams(η, κ, κh, nκh, ugrid, vgrid)
end

ConstDiffSteadyFlowParams(η, κ, κh, nκh, u::Array{Float64,2}, v::Array{Float64,2},
                          g::TwoDGrid) = ConstDiffSteadyFlowParams(η, κ, κh, nκh, u, v)

ConstDiffSteadyFlowParams(η, κ, u, v, g) = ConstDiffSteadyFlowParams(η, κ, 0, 0, u, v, g)


"""
Initialize an equation with constant diffusivity problem parameters p
and on a grid g.
"""
function Equation(p::ConstDiffParams, g::TwoDGrid)
  LC = zeros(g.Kr)
  @. LC = -p.η*g.kr^2 - p.κ*g.l^2
  FourierFlows.Equation{typeof(LC[1, 1]),2}(LC, calcN!)
end

function Equation(p::ConstDiffSteadyFlowParams, g::TwoDGrid)
  LC = zeros(g.Kr)
  @. LC = -p.η*g.kr^2 - p.κ*g.l^2 - p.κh*g.KKrsq^p.nκh
  FourierFlows.Equation{typeof(LC[1, 1]),2}(LC, calcN_steadyflow!)
end


# Vars
struct Vars <: AbstractVars
  c::Array{Float64,2}
  cx::Array{Float64,2}
  cy::Array{Float64,2}
  ch::Array{Complex{Float64},2}
  cxh::Array{Complex{Float64},2}
  cyh::Array{Complex{Float64},2}
end

""" Initialize the vars type on a grid g with zero'd arrays and t=0. """
function Vars(g::TwoDGrid)
  @createarrays Float64 (g.nx, g.ny) c cx cy
  @createarrays Complex{Float64} (g.nkr, g.nl) ch cxh cyh
  Vars(c, cx, cy, ch, cxh, cyh)
end




# Solvers ---------------------------------------------------------------------
"""
Calculate the advective terms for a tracer equation with constant
diffusivity.
"""
function calcN!(
  N::Array{Complex{Float64},2}, sol::Array{Complex{Float64},2},
  t::Float64, s::State, v::Vars, p::ConstDiffParams, g::TwoDGrid)

  v.ch .= sol
  @. v.cxh = im*g.kr*v.ch
  @. v.cyh = im*g.l*v.ch

  A_mul_B!(v.cx, g.irfftplan, v.cxh) # destroys v.cxh when using fftw
  A_mul_B!(v.cy, g.irfftplan, v.cyh) # destroys v.cyh when using fftw

  @. v.cx = -p.u(g.X, g.Y, s.t)*v.cx - p.v(g.X, g.Y, s.t)*v.cy # copies over v.cx
  A_mul_B!(N, g.rfftplan, v.cx)
  nothing
end


"""
Calculate the advective terms for a tracer equation with constant
diffusivity and time-constant flow.
"""
function calcN_steadyflow!(
  N::Array{Complex{Float64},2}, sol::Array{Complex{Float64},2},
  t::Float64, s::State, v::Vars, p::ConstDiffSteadyFlowParams, g::TwoDGrid)

  v.ch .= sol
  @. v.cxh = im*g.kr*v.ch
  @. v.cyh = im*g.l*v.ch

  A_mul_B!(v.cx, g.irfftplan, v.cxh) # destroys v.cxh when using fftw
  A_mul_B!(v.cy, g.irfftplan, v.cyh) # destroys v.cyh when using fftw

  @. v.cx = -p.u*v.cx - p.v*v.cy # copies over v.cx
  A_mul_B!(N, g.rfftplan, v.cx)
  nothing
end



# Helper functions ------------------------------------------------------------
""" Update state variables. """
function updatevars!(s::State, v::AbstractVars, g::TwoDGrid)
  v.ch .= s.sol
  ch1 = deepcopy(v.ch)
  A_mul_B!(v.c, g.irfftplan, ch1)
  nothing
end

updatevars!(prob::AbstractProblem) = updatevars!(prob.state, prob.vars, prob.grid)


""" Set the concentration field of the model with an array. """
function set_c!(s::State, v::AbstractVars, g::TwoDGrid, c::Array{Float64, 2})
  A_mul_B!(s.sol, g.rfftplan, c)
  updatevars!(s, v, g)
  nothing
end


""" Set the concentration field of the model with a function. """
function set_c!(s::State, v::AbstractVars, g::TwoDGrid, c::Function)
  cgrid = c.(g.X, g.Y)
  A_mul_B!(s.sol, g.rfftplan, cgrid)
  updatevars!(s, v, g)
  nothing
end

set_c!(prob::AbstractProblem, c) = set_c!(prob.state, prob.vars, prob.grid, c)


end # module
