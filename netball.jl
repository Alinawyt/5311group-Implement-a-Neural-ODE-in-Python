using Lux
import Random
import Optimisers
import Zygote
using ComponentArrays
using DiffEqFlux
using CSV
using DataFrames
using DifferentialEquations
using Plots
# Load data
data = CSV.read("./netball.csv", DataFrame)
dudt_model = Lux.Chain(
  WrappedFunction(u -> u[3:4]),
  Lux.Parallel(vcat,
# extract (ẋ, ẏ) from input (x, y, ẋ, ẏ)
# output is concatenation of:
#   inputs (ẋ, ẏ)
#   neural network a(ẋ, ẏ)
NoOpLayer(),
Lux.Chain(
      Lux.Dense(2 => 16, tanh),
      Lux.Dense(16 => 16, tanh),
      Lux.Dense(16 => 16, tanh),
      Lux.Dense(16 => 2)
) )
)
ps, st = Lux.setup(Random.default_rng(), dudt_model)
ps = ComponentArray(ps)

u0 = [data.x[1], data.y[1], data.v[1]*cosd(data.theta[1]), data.v[1]*sind(data.theta[1])]
xy_training = collect(Array(data[!,[:x,:y]])')

function L(ps)
  xy_predicted = Array(prob(u0, ps, st)[1])[1:2,:]
  sum(abs2, xy_predicted .- xy_training)
end

opt = Optimisers.setup(Optimisers.Adam(0.01f0), ps)
for epoch ∈ 1:1000#5000
  if rem(epoch, 10) == 0
    println("epoch $(epoch): Loss = $(L(ps))")
  end
  gs = Zygote.gradient(L, ps)[1]            # compute gradient wrt ps
  opt, ps = Optimisers.update(opt, ps, gs)  # update ps using the gradient
end

# NeuralODE
println("Execution time for Julia(DiffEqFlux.NeuralODE) implememt Neural ODEs:")
@time begin
prob = NeuralODE(dudt_model, (data.t[1], data.t[end]), Tsit5(); saveat=data.t)
end


xy_predicted=Array(prob(u0, ps, st)[1])[1:2,:]
x_predicted = xy_predicted[1, :]
y_predicted = xy_predicted[2, :]

# Plot the trajectories
plot(
    x_predicted, y_predicted, label="Predicted trajectory",
    xlabel="x_predicted", ylabel="y_predicted", title="Netball trajectory (Implementing with Julia)", legend=:topleft
)
plot!(
    data.x, data.y, label="True trajectory",
    xlabel="x", ylabel="y"
)
