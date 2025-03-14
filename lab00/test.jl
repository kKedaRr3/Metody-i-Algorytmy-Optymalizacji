using LinearAlgebra
using Plots
using Random

function f(x, θ)
    powers = x.^(0:(length(θ)-1))
    return dot(powers, θ)
end

ts = -1.0:0.01:2.0
params = [2.0, 3.0, 1.0, -1.5]
ys = [f(t, params) for t in ts]
plot(ts, ys)


Random.seed!(15)
x_data = sort(randn(20).*3)
θ_data = [0.2, -8.0, -5.0, 1]
y_data = [f(x, θ_data) + randn()*4.0 for x in x_data]
scatter(x_data, y_data; label="dane")
t_plot = -5:0.01:5.5
plot!(t_plot, [f(x, θ_data) for x in t_plot]; label="oryginalna krzywa")



struct CostF
    cost_x::Vector{Float64}
    cost_y::Vector{Float64}
end
(cf::CostF)(θ) = sum((map(x -> f(x, θ), cf.cost_x) - y_data).^2)

struct GradF
    cost_x::Vector{Float64}
    cost_y::Vector{Float64}
end
function (gf::GradF)(storage, θ)
    storage .= 0
    for i in eachindex(gf.cost_x)
        x = gf.cost_x[i]
        powers = x.^(0:(length(θ)-1))
        storage .+= 2 .* (dot(powers, θ) - gf.cost_y[i]) .* powers
    end
    return storage
end

cost_data = CostF(x_data, y_data)
grad_data = GradF(x_data, y_data)

using Optim
using LineSearches

println(cost_data(randn(4)))


optim_res = optimize(cost_data, grad_data, randn(4), ConjugateGradient())

println("Znalezione wartości: $(optim_res.minimizer)")
println("Funkja kosztu dla znalezionych: $(optim_res.minimum)")
println("Oryginalne wartości:  $(θ_data)")
println("Funkcja kosztu dla oryginalnych wartości: $(cost_data(θ_data))")

scatter(x_data, y_data; label="dane")
t_plot = -5:0.01:5.5
plot!(t_plot, [f(x, θ_data) for x in t_plot]; label="oryginalna krzywa")
plot!(t_plot, [f(x, optim_res.minimizer) for x in t_plot]; label="optymalna krzywa")


# Funkcja kosztu (funkcja kwadratowa dla regresji liniowej)
function cost_function(X, y, θ)
    m = length(y)
    predictions = X * θ
    cost = (1 / (2 * m)) * sum((predictions - y) .^ 2)
    return cost
end

# Gradient funkcji kosztu
function compute_gradient(X, y, θ)
    m = length(y)
    predictions = X * θ
    gradient = (1 / m) * (X' * (predictions - y))
    return gradient
end

# Metoda największego spadku ze stałym krokiem
function gradient_descent_fixed_step(X, y, α, iterations)
    θ = zeros(size(X, 2))
    cost_history = []

    for i in 1:iterations
        gradient = compute_gradient(X, y, θ)
        θ -= α * gradient
        push!(cost_history, cost_function(X, y, θ))
    end

    return θ, cost_history
end

# Metoda największego spadku z krokiem zanikającym
function gradient_descent_decay_step(X, y, γ, iterations)
    θ = zeros(size(X, 2))
    cost_history = []
    α = 1.0  # Zaczynamy od kroku 1

    for i in 1:iterations
        gradient = compute_gradient(X, y, θ)
        θ -= α * gradient
        push!(cost_history, cost_function(X, y, θ))
        α *= γ  # Zmniejszamy krok zgodnie ze wzorem
    end

    return θ, cost_history
end

# Generowanie danych do testów
Random.seed!(42)
m = 100
X = 2 * rand(m, 1)
y = 4 .+ 3 * X .+ randn(m)

# Dodanie kolumny jedynek dla wyrazu wolnego
X_b = hcat(ones(m), X)

# Parametry
α_fixed = 0.1
γ = 0.9
iterations = 50

# Uruchamianie algorytmów
θ_fixed, cost_fixed = gradient_descent_fixed_step(X_b, y, α_fixed, iterations)
θ_decay, cost_decay = gradient_descent_decay_step(X_b, y, γ, iterations)

# Wyniki
println("Parametry (stały krok): ", θ_fixed)
println("Parametry (krok zanikający): ", θ_decay)

# Wykres funkcji kosztu
plot(1:iterations, cost_fixed, label="Stały krok (α = 0.1)", xlabel="Iteracje", ylabel="Funkcja kosztu", legend=:topright)
plot!(1:iterations, cost_decay, label="Krok zanikający (γ = 0.9)")

using Manopt, Manifolds
(cf::CostF)(::Euclidean, θ) = cf(θ)
(gf::GradF)(::Euclidean, θ) = gf(similar(θ), θ)
check_gradient(Euclidean(4), cost_data, grad_data, plot=true)