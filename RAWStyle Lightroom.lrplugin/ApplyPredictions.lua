-- Apply supervised RAWStyle predictions to selected Lightroom Classic photos.
-- This plug-in never opens / edits the .lrcat SQLite file directly.
-- Install through File -> Plug-in Manager -> Add.
local LrApplication = import 'LrApplication'
local LrDialogs = import 'LrDialogs'
local LrTasks = import 'LrTasks'
local LrFileUtils = import 'LrFileUtils'
local LrPathUtils = import 'LrPathUtils'

local allowed = {
    Temperature = true, Tint = true,
    Exposure2012 = true, Contrast2012 = true,
    Highlights2012 = true, Shadows2012 = true,
    Whites2012 = true, Blacks2012 = true,
    Texture = true, Clarity2012 = true, Dehaze = true,
    Vibrance = true, Saturation = true, Sharpness = true,
    SharpenDetail = true, SharpenEdgeMasking = true,
    ColorNoiseReduction = true, LuminanceSmoothing = true,
    PostCropVignetteAmount = true,
}

local function unescape(text)
    return (text:gsub("%%(%x%x)", function(hex)
        return string.char(tonumber(hex, 16))
    end))
end

local function columns(line)
    local fields = {}
    for cell in (line .. "\t"):gmatch("(.-)\t") do
        fields[#fields + 1] = cell
    end
    return fields
end

local function normal(path)
    if not path then return nil end
    return LrPathUtils.standardizePath(path):lower()
end

local function readPredictions(content)
    local lines = {}
    for line in content:gmatch("[^\r\n]+") do
        lines[#lines + 1] = line
    end
    if lines[1] ~= "#RAWSTYLE\t2" then
        error("Not a RAWStyle v2 predictions file.")
    end
    if not lines[2] then error("Missing predictions header.") end
    local keys = columns(lines[2])
    if keys[1] ~= "path" then error("Missing path column.") end
    local byPath = {}
    local count = 0
    for lineIndex = 3, #lines do
        local cells = columns(lines[lineIndex])
        if cells[1] and cells[1] ~= "" then
            local filePath = normal(unescape(cells[1]))
            if filePath and byPath[filePath] == nil then
                local settings = {}
                for idx = 2, #keys do
                    local key = keys[idx]
                    if allowed[key] then
                        local val = tonumber(cells[idx])
                        if val then settings[key] = val end
                    end
                end
                if settings.Temperature and settings.Tint then
                    -- Necessary for Lightroom to apply predicted Kelvin/Tint.
                    settings.WhiteBalance = "Custom"
                    byPath[filePath] = settings
                    count = count + 1
                end
            end
        end
    end
    return byPath, count
end

LrTasks.startAsyncTask(function()
    local chosen = LrDialogs.runOpenPanel {
        title = "Select RAWStyle Predictions",
        prompt = "Use Predictions",
        canChooseFiles = true,
        canChooseDirectories = false,
        allowsMultipleSelection = false,
        fileTypes = { "rawstyle" },
    }
    if not chosen or not chosen[1] then return end
    local ok, result, count = pcall(function()
        local data = LrFileUtils.readFile(chosen[1])
        if not data then error("Could not read that file.") end
        return readPredictions(data)
    end)
    if not ok then
        LrDialogs.showError("RAWStyle could not read predictions: " .. tostring(result))
        return
    end
    local predictions = result
    local catalog = LrApplication.activeCatalog()
    local selected = catalog:getTargetPhotos()
    local matches = {}
    for _, photo in ipairs(selected) do
        local photoPath = normal(photo:getRawMetadata("path"))
        if photoPath and predictions[photoPath] then
            matches[#matches + 1] = { photo = photo, settings = predictions[photoPath] }
        end
    end
    if #matches == 0 then
        LrDialogs.message("No matching selected RAW photos",
            "Import your new RAW shoot into Lightroom Classic, select the photos " ..
            "you want to edit, and ensure the files are still in their original folder.", "warning")
        return
    end
    local confirmation = LrDialogs.confirm(
        "Apply RAWStyle to " .. tostring(#matches) .. " photos?",
        "Apply predicted adjustments, including Temperature and Tint, to the selected photos.",
        "Apply", "Cancel")
    if confirmation ~= "ok" then return end

    local succeeded, changed, errors = pcall(function()
        local done = 0
        local failures = {}
        catalog:withWriteAccessDo("Apply RAWStyle predictions", function()
            for _, item in ipairs(matches) do
                local success, err = pcall(function()
                    item.photo:applyDevelopSettings(item.settings, "RAWStyle Prediction")
                end)
                if success then
                    done = done + 1
                else
                    failures[#failures + 1] = tostring(err)
                end
            end
        end)
        return done, failures
    end)
    if not succeeded then
        LrDialogs.showError("Could not apply edits: " .. tostring(changed))
        return
    end
    if #errors > 0 then
        LrDialogs.message("Some predictions could not be applied",
            tostring(changed) .. " edited; " .. tostring(#errors) .. " failed. " ..
            "First error: " .. errors[1], "warning")
    else
        LrDialogs.message("RAWStyle finished",
            tostring(changed) .. " Lightroom photos updated.", "info")
    end
end)
