function varargout = JST(varargin)
% JST MATLAB code for JST.fig
%      JST, by itself, creates a new JST or raises the existing
%      singleton*.
%
%      H = JST returns the handle to a new JST or the handle to
%      the existing singleton*.
%
%      JST('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in JST.M with the given input arguments.
%
%      JST('Property','Value',...) creates a new JST or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before JST_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to JST_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help JST

% Last Modified by GUIDE v2.5 24-Jul-2017 19:23:36

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @JST_OpeningFcn, ...
                   'gui_OutputFcn',  @JST_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before JST is made visible.
function JST_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to JST (see VARARGIN)

% Choose default command line output for JST
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes JST wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = JST_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



function display_Callback(hObject, eventdata, handles)
% hObject    handle to display (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of display as text
%        str2double(get(hObject,'String')) returns contents of display as a double


% --- Executes during object creation, after setting all properties.
function display_CreateFcn(hObject, eventdata, handles)
% hObject    handle to display (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Browse_File.
function Browse_File_Callback(hObject, eventdata, handles)
% hObject    handle to Browse_File (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[namafile alamatfile]=uigetfile({'*.xlsx';'*.xls'},'Buka File');
data = xlsread(namafile,'x');
save data data;
%% Alamat File Handle Oleh Display
set(handles.display,'String',[alamatfile namafile])

%% Data Uji Masuk ke Table1
set(handles.table1,'Data',data)
handles.data = data
guidata(hObject, handles)

function batas_Callback(hObject, eventdata, handles)
% hObject    handle to batas (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of batas as text
%        str2double(get(hObject,'String')) returns contents of batas as a double

% --- Executes during object creation, after setting all properties.
function batas_CreateFcn(hObject, eventdata, handles)
% hObject    handle to batas (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in input_batas.
function input_batas_Callback(hObject, eventdata, handles)
% hObject    handle to input_batas (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%% memasukkan parameter========================================================================
batas =str2double(get(findobj(gcf,'Tag','batas'),'String'));

%% Menghitung Autokorelasi ====================================================================
data = handles.data;
[HitungAutokorelasi hasilinput]= autokorelasi(data,batas);
save hasilinput hasilinput;
save HitungAutokorelasi HitungAutokorelasi;
set(handles.table2,'Data',HitungAutokorelasi)
handles.input = hasilinput;
%Normalisasi data input====================================================================
[hasilNorm nilaiMax nilaiMin] = normalisasi(hasilinput);
handles.nilaiMax=nilaiMax;
handles.nilaiMin=nilaiMin;
set(handles.table3,'Data',hasilNorm);

%Nilai Jumlah Input====================================================================
[bar kol] = size(HitungAutokorelasi);
handles.jumlah_input = bar;
set(findobj(gcf,'Tag','jumlah_input'),'String',bar);
% set(handles.jumlah_input,'Data',bar)
%% Handles Hasil Normalisasi
handles.hasilNorm = hasilNorm
guidata(hObject, handles)


function jumlah_hidden_layer_Callback(hObject, eventdata, handles)
% hObject    handle to jumlah_hidden_layer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of jumlah_hidden_layer as text
%        str2double(get(hObject,'String')) returns contents of jumlah_hidden_layer as a double


% --- Executes during object creation, after setting all properties.
function jumlah_hidden_layer_CreateFcn(hObject, eventdata, handles)
% hObject    handle to jumlah_hidden_layer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


function maximum_epoch_Callback(hObject, eventdata, handles)
% hObject    handle to maximum_epoch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of maximum_epoch as text
%        str2double(get(hObject,'String')) returns contents of maximum_epoch as a double


% --- Executes during object creation, after setting all properties.
function maximum_epoch_CreateFcn(hObject, eventdata, handles)
% hObject    handle to maximum_epoch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function alfa_Callback(hObject, eventdata, handles)
% hObject    handle to alfa (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of alfa as text
%        str2double(get(hObject,'String')) returns contents of alfa as a double


% --- Executes during object creation, after setting all properties.
function alfa_CreateFcn(hObject, eventdata, handles)
% hObject    handle to alfa (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function momentum_Callback(hObject, eventdata, handles)
% hObject    handle to alfa (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of alfa as text
%        str2double(get(hObject,'String')) returns contents of alfa as a double


% --- Executes during object creation, after setting all properties.
function momentum_CreateFcn(hObject, eventdata, handles)
% hObject    handle to alfa (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function goal_Callback(hObject, eventdata, handles)
% hObject    handle to goal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of goal as text
%        str2double(get(hObject,'String')) returns contents of goal as a double


% --- Executes during object creation, after setting all properties.
function goal_CreateFcn(hObject, eventdata, handles)
% hObject    handle to goal (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function jumlah_input_Callback(hObject, eventdata, handles)
% hObject    handle to jumlah_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of jumlah_input as text
%        str2double(get(hObject,'String')) returns contents of jumlah_input as a double


% --- Executes during object creation, after setting all properties.
function jumlah_input_CreateFcn(hObject, eventdata, handles)
% hObject    handle to jumlah_input (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in Hitung.
function Hitung_Callback(hObject, eventdata, handles)
% hObject    handle to Hitung (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%% Mengambil Data Input
jumlah_hidden_layer =str2double(get(findobj(gcf,'Tag','jumlah_hidden_layer'),'String'));
maximum_epoch =str2double(get(findobj(gcf,'Tag','maximum_epoch'),'String'));
momentum =str2double(get(findobj(gcf,'Tag','momentum'),'String'));
alfa =str2double(get(findobj(gcf,'Tag','alfa'),'String'));
goal =str2double(get(findobj(gcf,'Tag','goal'),'String'));
% kol =str2double(get(findobj(gcf,'Tag','goal'),'String'));
jumlah_input = handles.jumlah_input;
training = handles.training;
testing = handles.testing;
testingAct=handles.testingAct;
nilaiMax=handles.nilaiMax;
nilaiMin=handles.nilaiMin;
% % testingTarget=handles.testingTarget;
%% Proses Backpro
%% AmbilBias
load('BobotV/vBobot.mat');
load('BobotW/wBobot.mat');

hasil_in = vBobot{jumlah_hidden_layer,jumlah_input};
hasil_out = wBobot{jumlah_hidden_layer,1};
%% Proses Backpro
[biasW biasV MSE niloop] = Backpro(training,hasil_in,hasil_out,jumlah_hidden_layer,alfa,maximum_epoch,momentum,goal);

[bar1 kol1] = size(MSE);
%% Testing ke Data Training
[hasilTest]=protest(testing,jumlah_hidden_layer,biasV,biasW);
handles.hasilTest=hasilTest;
%% Hasil Denormalisasi
%  [hasilDenorm]=denormalisasi(hasilTest,nilaiMax,nilaiMin);
[hasilDenorm]=denormalisasi2(hasilTest,testingAct);
handles.hasilDenorm=hasilDenorm;

%% Untuk Plot Grafik
[a b] = size(testingAct);
[c d] = size(hasilDenorm);
handles.a = a
handles.c = c
handles.MSE = MSE
handles.niloop = niloop

%%
[bb kk]=size(testing);
mseakhir=0;
finish=[];
mape=0;
for x=1:c
    actual = testingAct(x,1);
    prediksi = hasilDenorm(x,1);
    selisih = abs(actual-prediksi);
    sel = (hasilTest(x,:)-testing(x,kk));
    presentase = (selisih/prediksi)*100;
    finish=[finish; actual prediksi selisih presentase];
    mape=mape+presentase;
    mseakhir = mseakhir + (sel*sel);
end

set(handles.table9,'Data',finish);

%% MAPE
% [hasil] = HitungMape(hasilDenorm, testingTarget);
hasilAkhir=(mape/c);
makhir =(mseakhir/c);
set(findobj(gcf,'Tag','mape'),'String',hasilAkhir);
set(findobj(gcf,'Tag','mseakhir'),'String',makhir);
guidata(hObject, handles)

function skenario_Callback(hObject, eventdata, handles)
% hObject    handle to skenario (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of skenario as text
%        str2double(get(hObject,'String')) returns contents of skenario as a double


% --- Executes during object creation, after setting all properties.
function skenario_CreateFcn(hObject, eventdata, handles)
% hObject    handle to skenario (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in K_Fold.
function K_Fold_Callback(hObject, eventdata, handles)
% hObject    handle to K_Fold (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%% Ambil Data Skenario
skenario =str2double(get(findobj(gcf,'Tag','skenario'),'String'));
input = handles.input;
%% Mencari Nilai K-Fold
hasilNorm=handles.hasilNorm

[hasilCross]=crossVal(hasilNorm);
[bar kol] = size(input);
testing = hasilNorm(hasilCross(skenario,1):hasilCross(skenario,2),:);
testingAct = input(hasilCross(skenario,1):hasilCross(skenario,2),kol);

training=[];
if(skenario==1)
    training=[training;hasilNorm(hasilCross(2,1):hasilCross(2,2),:)];
    training=[training;hasilNorm(hasilCross(3,1):hasilCross(3,2),:)];
else
    if(skenario==2)
        training=[training;hasilNorm(hasilCross(1,1):hasilCross(1,2),:)];
        training=[training;hasilNorm(hasilCross(3,1):hasilCross(3,2),:)];
    else
        training=[training;hasilNorm(hasilCross(1,1):hasilCross(1,2),:)];
        training=[training;hasilNorm(hasilCross(2,1):hasilCross(2,2),:)];
    end
end

%% Masukkan ke Tabel Nilai Training dan Testing
set(handles.table5,'Data',training)
set(handles.table6,'Data',testing)

%% Handles Data Training dan Testing
%% Handles Hasil Normalisasi
handles.training = training
%% Handles Hasil Normalisasi
handles.testing = testing
handles.testingAct = testingAct

guidata(hObject, handles)



function edit11_Callback(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit11 as text
%        str2double(get(hObject,'String')) returns contents of edit11 as a double


% --- Executes during object creation, after setting all properties.
function edit11_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit11 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

function mape_Callback(hObject, eventdata, handles)
% hObject    handle to mape (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of mape as text
%        str2double(get(hObject,'String')) returns contents of mape as a double


% --- Executes during object creation, after setting all properties.
function mape_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mape (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in D_Training.
function D_Training_Callback(hObject, eventdata, handles)
% hObject    handle to D_Training (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
figure
plot ([1:handles.niloop]',handles.MSE','Color',[0,0.7,0.9])
title('2-D Line Plot')
xlabel('Loop MaxEpoch')
ylabel('Nilai MSE')

% --- Executes on button press in D_Testing.
function D_Testing_Callback(hObject, eventdata, handles)
% hObject    handle to D_Testing (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%plot target vs hasil testing ==========================================
figure
h = plot ([1:handles.a]',handles.testingAct','c*',[1:handles.c]',handles.hasilDenorm','bo')
set(h(1),'Color',[1,0,0])
title('Hasil pengujian dengan target pengujian : Target (o), Output (*)')
 xlabel('Data ke-')
  ylabel('Target/Output')



function mseakhir_Callback(hObject, eventdata, handles)
% hObject    handle to mseakhir (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of mseakhir as text
%        str2double(get(hObject,'String')) returns contents of mseakhir as a double


% --- Executes during object creation, after setting all properties.
function mseakhir_CreateFcn(hObject, eventdata, handles)
% hObject    handle to mseakhir (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
