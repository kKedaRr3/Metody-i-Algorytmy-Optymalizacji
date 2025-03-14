using Random
using Distributions
using Plots
using LinearAlgebra

function gradient_descent_fixed_step(x, y, α, iterations, fun, gradient)
    cost_history = []
    trajectory = [(x, y)]  # Zapisujemy trajektorię punktu

    for i in 1:iterations
        gradient_x, gradient_y = gradient(x, y)
        x -= α * gradient_x
        y -= α * gradient_y
        push!(cost_history, fun(x, y))
        push!(trajectory, (x, y))  # Dodajemy kolejne punkty trajektorii
    end

    return x, y, cost_history, trajectory
end

function himmelblau(x, y)
    return (x^2 + y - 11)^2 + (x + y^2 - 7)^2
end

# Gradient funkcji Himmelblaua
function himmelblau_gradient(x, y)
    df_dx = 4 * x * (x^2 + y - 11) + 2 * (x + y^2 - 7)
    df_dy = 2 * (x^2 + y - 11) + 4 * y * (x + y^2 - 7)
    return df_dx, df_dy
end

Random.seed!(42)
num_points = 10
start_points = [(rand(Uniform(-5, 5)), rand(Uniform(-5, 5))) for _ in 1:num_points]

all_cost_histories = []
α_values = [0.001, 0.01, 0.025]
iterations = 100

for α in α_values
    x_vals = []
    y_vals = []
    z_vals = []
    trajectories = []

    for (x0, y0) in start_points
        x_opt, y_opt, cost_history, trajectory = gradient_descent_fixed_step(x0, y0, α, iterations, himmelblau, himmelblau_gradient)
        push!(all_cost_histories, cost_history)        
        push!(x_vals, x0)
        push!(y_vals, y0)
        push!(z_vals, cost_history[end])  # Ostateczna wartość funkcji kosztu
        push!(trajectories, trajectory)  # Pełna trajektoria

    end

    # Wykres konturowy funkcji Himmelblaua
    x_range = range(-5, 5, length=100)
    y_range = range(-5, 5, length=100)
    z_matrix = [himmelblau(x, y) for x in x_range, y in y_range]

    contour_plot = contourf(x_range, y_range, z_matrix, fill=true, color=:plasma, title="Trajektorie gradientu dla α = $α")

    # Dodanie trajektorii gradientowego spadku do wykresu konturowego
    for traj in trajectories
        plot!(contour_plot, [p[1] for p in traj], [p[2] for p in traj], linewidth=2, color=:white, arrow=true)
    end

    display(contour_plot)  # Poprawne wyświetlenie wykresu
    global all_cost_histories = []
end
