using Random
using Distributions
using Plots
using LinearAlgebra

# Optymalizacja z fixed steo
function gradient_descent_fixed_step(x, y, α, iterations, fun, gradient)
    cost_history = []

    for i in 1:iterations
        gradient_x, gradient_y = gradient(x, y)
        x -= α * gradient_x
        y -= α * gradient_y
        push!(cost_history, fun(x, y))
    end

    return x, y, cost_history
end

# Funkcja Armijo
function armijo_step_size(x, y, fun, gradient, β=0.5, c=0.1)
    α = 1  
    α_min = 0
    f_old = fun(x, y)
    grad_x, grad_y = gradient(x, y)
    
    while fun(x - α * grad_x, y - α * grad_y) > f_old - c * α * (grad_x^2 + grad_y^2) && α > α_min
        α *= β  
    end
    
    return α
end

# Optymalizacja z metoda armijo
function gradient_descent_armijo(x, y, iterations, fun, gradient, β=0.5, c=0.1)
    cost_history = []

    for i in 1:iterations
        grad_x, grad_y = gradient(x, y)

        α = armijo_step_size(x, y, fun, gradient, β, c)

        x -= α * grad_x
        y -= α * grad_y
        push!(cost_history, fun(x, y))
    end

    return x, y, cost_history
end

# Funkcja Himmelblaua
function himmelblau(x, y)
    return (x^2 + y - 11)^2 + (x + y^2 - 7)^2
end

# Gradient funkcji Himmelblaua
function himmelblau_gradient(x, y)
    df_dx = 4 * x * (x^2 + y - 11) + 2 * (x + y^2 - 7)
    df_dy = 2 * (x^2 + y - 11) + 4 * y * (x + y^2 - 7)
    return df_dx, df_dy
end


# Wyniki fixed step
Random.seed!(42)
num_points = 10
start_points = [(rand(Uniform(-5, 5)), rand(Uniform(-5, 5))) for _ in 1:num_points]

all_cost_histories = []
α_values = [0.001, 0.01, 0.025]
iterations = 100

for α in α_values
    for (x0, y0) in start_points
        x_opt, y_opt, cost_history = gradient_descent_fixed_step(x0, y0, α, iterations, himmelblau, himmelblau_gradient)
        push!(all_cost_histories, cost_history)        
    end

    local plt = plot(1:iterations, all_cost_histories[1], label="Punkt 1", xlabel="Iteracja", ylabel="Wartość funkcji", title="Zależność wartości funkcji od iteracji")
    for i in 2:num_points
        plot!(plt, 1:iterations, all_cost_histories[i], label="Punkt $i")
    end
    display(plt)
    global all_cost_histories = []
end


armijo_params = [
    (0.9, 0.05),  
    (0.9, 0.1),  
    (0.7, 0.05),  
    (0.7, 0.1),
    (0.5, 0.05),  
    (0.5, 0.1)
]


# Wyniki armijo
armijo_results = Dict()

for (β, c) in armijo_params
    avg_costs = zeros(iterations)

    for (x0, y0) in start_points
        _, _, cost_history = gradient_descent_armijo(x0, y0, iterations, himmelblau, himmelblau_gradient, β, c)
        avg_costs += cost_history
    end

    avg_costs /= num_points
    armijo_results[(β, c)] = avg_costs
end

plt = plot()

for ((β, c), costs) in armijo_results
    if !isempty(costs)
        plot!(plt, 1:iterations, costs, label="Armijo (β=$β, c=$c)", lw=2)
    else
        println("Brak danych dla Armijo (β=$β, c=$c)")
    end
end

xlabel!("Iteracja")
ylabel!("Średnia wartość funkcji")
title!("Wyniki metody Armijo")
display(plt)


# Porownanie wynikow armijo i fixed step
plt = plot()
fixed_step_results = Dict()

for α in α_values
    avg_costs = zeros(iterations)
    
    for (x0, y0) in start_points
        _, _, cost_history = gradient_descent_fixed_step(x0, y0, α, iterations, himmelblau, himmelblau_gradient)
        avg_costs += cost_history
    end
    
    avg_costs /= num_points
    fixed_step_results[α] = avg_costs
end

for (α, costs) in fixed_step_results
    plot!(plt, 1:iterations, costs, label="Stały krok α = $α", lw=2, linestyle=:dot)
end

for ((β, c), costs) in armijo_results
    if !isempty(costs)
        plot!(plt, 1:iterations, costs, label="Armijo (β=$β, c=$c)", lw=2)
    end
end

xlabel!("Iteracja")
ylabel!("Średnia wartość funkcji")
title!("Porównanie zbieżności: Armijo vs. stały krok")

display(plt)