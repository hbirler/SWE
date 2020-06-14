target_sources(swe 
    PRIVATE
    Logger.cpp
        DeviceFloat2D.cpp

    PUBLIC
        DeviceFloat2D.hh
    Logger.hh
    ProgressBar.hh
    args.hh
    help.hh
)
