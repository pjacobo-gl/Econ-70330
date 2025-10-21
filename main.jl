using DataFrames, CSV

# Import data
df = DataFrame(load("us_data_updated.dta"))

# 1. Keep only selected columns
df = df[:, [:timeseq, :year, :quarter, :fehor, :ind_fe, :ind_fr, :ind_for]]

# 2. Reshape wide (pivot_wider)
df_wide = unstack(df, [:timeseq], :fehor, [:ind_fe, :ind_fr, :ind_for])

# 3. Drop columns if they exist
for col in [:ind_for_30, :ind_for_41]
    if hasproperty(df_wide, col)
        select!(df_wide, Not(col))
    end
end

# 4. Generate new columns
df_wide.ind_for_30 = ((df_wide.ind_for0 ./ 400 .+ 1) .* (df_wide.ind_for1 ./ 400 .+ 1) .* (df_wide.ind_for2 ./ 400 .+ 1) .* (df_wide.ind_for3 ./ 400 .+ 1) .- 1) .* 100
df_wide.ind_for_41 = ((df_wide.ind_for1 ./ 400 .+ 1) .* (df_wide.ind_for2 ./ 400 .+ 1) .* (df_wide.ind_for3 ./ 400 .+ 1) .* (df_wide.ind_for4 ./ 400 .+ 1) .- 1) .* 100

df_wide.ind_fe_30 = ((df_wide.ind_fe0 ./ 400 .+ 1) .* (df_wide.ind_fe1 ./ 400 .+ 1) .* (df_wide.ind_fe2 ./ 400 .+ 1) .* (df_wide.ind_fe3 ./ 400 .+ 1) .- 1) .* 100

# 5. fr_30_41 = ind_for_30 - lag(ind_for_41)
df_wide.fr_30_41 = df_wide.ind_for_30 .- [missing; df_wide.ind_for_41[1:end-1]]

# 6. Regression (using GLM.jl)
using GLM
reg1 = lm(@formula(ind_fe_30 ~ fr_30_41), df_wide)

# 7. IV regression (GMM not directly available, but you can use IV regression via FixedEffectModels.jl or similar)

# 8. Join with macro data
macro_df = CSV.read("Macro-data---FRED.csv", DataFrame)
df_joined = innerjoin(df_wide, macro_df, on = [:year, :quarter])

# 9. Generate lagged variables
function lag(x, n=1)
    return [fill(missing, n); x[1:end-n]]
end

df_joined.L_inf = lag(df_joined.inflation_realtime)
df_joined.L_tb3ms = lag(df_joined.tb3ms)
df_joined.L_unrate = lag(df_joined.unrate)
df_joined.oilP = coalesce.(df_joined.oilprice, df_joined.mcoilwtico)
df_joined.LD_oilP = log.(lag(df_joined.oilP) ./ lag(df_joined.oilP, 2))

df_joined.L_inf4 = df_joined.ind_for_30