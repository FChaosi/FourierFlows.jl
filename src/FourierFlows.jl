module FourierFlows

using Requires, FFTW, Statistics
import LinearAlgebra: mul!, ldiv!

export 
  AbstractGrid,
  AbstractParams,
  AbstractVars,
  AbstractEquation,
  AbstractTimeStepper,
  AbstractState,
  AbstractProblem,

  AbstractForwardEulerTimeStepper,
  AbstractFilteredForwardEulerTimeStepper,
  AbstractRK4TimeStepper,
  AbstractFilteredRK4TimeStepper

# --
# Abstract supertypes
# --

abstract type AbstractGrid end
abstract type AbstractParams end
abstract type AbstractVars end
abstract type AbstractTimeStepper end
abstract type AbstractEquation end
abstract type AbstractState end
abstract type AbstractProblem end


# --
# Base functionality
# --

include("problemstate.jl")
include("domains.jl")
include("diagnostics.jl")
include("output.jl")
include("utils.jl")
include("timesteppers.jl")


# --
# Physics
# --

include("physics/twodturb.jl")
include("physics/barotropicqg.jl")
include("physics/traceradvdiff.jl")
include("physics/kuramotosivashinsky.jl")
include("physics/verticallycosineboussinesq.jl")
include("physics/verticallyfourierboussinesq.jl")

# --
# CUDA/GPU functionality
# --

@require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
  using CuArrays
  include("cuda/cuutils.jl")
  include("cuda/cuproblemstate.jl")
  include("cuda/cudomains.jl")
  include("cuda/cutimesteppers.jl")
end

end # module
