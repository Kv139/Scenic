
$ports = @(20001,20002)
$worldPath = "C:\Users\ksv14\Downloads\TEMP\numpy_test\numpy_test\Scenic\examples\webots\vacuum\worlds\create2.wbt"
$webotsPath = "C:\Users\ksv14\AppData\Local\Programs\Webots\msys64\mingw64\bin\webotsw.exe"


# Start Webots instances (non-blocking)
foreach ($port in $ports){
    $env:WEBOTS_PORT = $port
    Start-Process -NoNewWindow -FilePath "$webotsPath" -ArgumentList "--port=$port", "$worldPath"
    Start-Sleep -Seconds 6  # wait a bit before starting the next one
}

# Give Webots more time to fully initialize before launching supervisors
Start-Sleep -Seconds 5

# Launch supervisors
foreach ($port in $ports){
    Start-Process -NoNewWindow -FilePath "python" -ArgumentList "scenic_supervisor.py", "--port=$port", "--robot-name=Supervisor"
}
