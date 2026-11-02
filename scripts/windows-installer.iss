#ifndef AppVersion
  #error AppVersion must be supplied by build_windows.ps1
#endif

[Setup]
AppId={{B8A24B86-83CB-4B5A-AE5F-E5D7B3D504BC}
AppName=LLMInsight
AppVersion={#AppVersion}
AppPublisher=LLMInsight contributors
AppPublisherURL=https://github.com/ai-v5/LLMInsight
AppSupportURL=https://github.com/ai-v5/LLMInsight/issues
DefaultDirName={localappdata}\Programs\LLMInsight
DefaultGroupName=LLMInsight
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename=LLMInsight-Setup-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\LLMInsight.exe
CloseApplications=yes

[Languages]
Name: "chinesesimp"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\build\windows-dist\LLMInsight\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\licenses\echarts-5.5.1\*"; DestDir: "{app}\licenses\echarts-5.5.1"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\LLMInsight"; Filename: "{app}\LLMInsight.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\LLMInsight"; Filename: "{app}\LLMInsight.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\LLMInsight.exe"; Description: "{cm:LaunchProgram,LLMInsight}"; Flags: nowait postinstall skipifsilent
