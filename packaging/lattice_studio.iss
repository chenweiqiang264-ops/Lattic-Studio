; Lattice Studio Windows installer.
; Build after the onedir package exists:
;   ISCC.exe packaging\\lattice_studio.iss

#define AppName "Lattice Studio"
#define AppVersion "0.1.1"
#define AppPublisher "Lattice Studio"
#define AppExeName "LatticeStudio.exe"
#define DistDir "..\\dist\\LatticeStudio"

[Setup]
AppId={{B9F42F2E-1EA4-4D2B-92C8-4F7F1A9F7D0A}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\\Programs\\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayIcon={app}\\{#AppExeName}
OutputDir=..\\dist\\installer
OutputBaseFilename=LatticeStudio-Setup-{#AppVersion}-x64
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
CloseApplications=yes
RestartApplications=no
Uninstallable=yes
SetupLogging=yes
VersionInfoVersion={#AppVersion}.0
VersionInfoDescription={#AppName} installer
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#DistDir}\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\\{#AppName}"; Filename: "{app}\\{#AppExeName}"
Name: "{autodesktop}\\{#AppName}"; Filename: "{app}\\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: dirifempty; Name: "{app}"
