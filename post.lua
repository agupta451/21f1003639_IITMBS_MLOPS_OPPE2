-- Read test.csv into a table
local lines = {}
-- Make sure test.csv is in the same directory where you run wrk
for line in io.lines("test.csv") do
    table.insert(lines, line)
end

-- Remove header
table.remove(lines, 1)

local headers = {"age", "gender", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"}

math.randomseed(os.time())

request = function()
    -- Pick a random line from the data
    local random_line = lines[math.random(1, #lines)]
    
    -- Split the CSV line into values
    local values = {}
    for value in string.gmatch(random_line, "([^,]+)") do
        table.insert(values, value)
    end

    -- Construct JSON payload
    local json_body = "{"
    for i, key in ipairs(headers) do
        local value = values[i]
        -- Check if the value is numeric or a string (for gender)
        if tonumber(value) then
            json_body = json_body .. '"' .. key .. '":' .. value
        else
            json_body = json_body .. '"' .. key .. '":"' .. value .. '"'
        end
        if i < #headers then
            json_body = json_body .. ","
        end
    end
    json_body = json_body .. "}"

    wrk.method = "POST"
    wrk.body = json_body
    wrk.headers["Content-Type"] = "application/json"
    
    return wrk.format(nil)
end
