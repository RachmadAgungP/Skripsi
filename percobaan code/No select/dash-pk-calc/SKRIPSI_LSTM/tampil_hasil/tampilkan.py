import pandas as pd
import numpy as np
def tampung_hitung_manual(jenis_proses,data):
    if jenis_proses == "forward": 
      var_hidden_layer = data
      list_table_hitung_str = ["I","Z","C","O","F","I_in","C_bar","h","W"]
    elif jenis_proses == "backward":
      var_hidden_layer = data
      list_table_hitung_str = ["dE_dW_t", "dE_dh_tminus1", "dE_dc_tminus1", "dE_do_t", "dE_dc_t", "dE_di_t", "dE_dcbar_t", "dE_df_t","dE_dzcbar_t","dE_dzi_t","dE_dzf_t","dE_dzo_t","dE_dz_t","dE_dI_t", "dE_dh_tplus1", "dE_dc_tplus1"]
    elif jenis_proses == "update bobot":
      var_hidden_layer = data
      list_table_hitung_str = ["new_dE_dW_t"]
    elif jenis_proses == "optimasi":
      var_hidden_layer = data
      list_table_hitung_str = ["optimasi_bobot"]
    else:
      var_hidden_layer = data
      list_table_hitung_str = ["data"]  

    data_full = {}
    data_perhitungan = pd.DataFrame(data_full) 
    vel_hidden_layer = []
    count = 0
    for i in var_hidden_layer:
        data_perhitungan.insert(count, list_table_hitung_str[count],np.transpose(np.transpose(i)).tolist(), True) 
        count += 1
    return data_perhitungan

def view_hitung_manual(jenis_proses,data_t,kon,urutan_ke):
    print()
    if jenis_proses == "forward":
        print ("PROSES FORWARD")
        if urutan_ke != "print full":
          print ("tabel hasil perhitungan %s urutan ke %s"%(kon,urutan_ke))
          if (kon == "I" or kon == "Z" ):
              view = pd.DataFrame(data=data_t.loc[urutan_ke,kon],index=["C","i","f","o"],columns=[urutan_ke])
          elif (kon =="W"):
              view = pd.DataFrame(data=data_t.loc[urutan_ke,kon],index=["C","i","f","o"], columns=["bias","date","x(close)","h(close)"])
          else:
              view = pd.DataFrame(data=data_t.loc[urutan_ke,kon],index=[kon],columns=[urutan_ke])
        else:
          print ("tabel hasil perhitungan %s urutan ke %s"%(kon,urutan_ke))
          if (kon == "I" or kon == "Z"):
            view = pd.DataFrame(data=data_t[kon].tolist(),columns=["C","i","f","o"]).T
          elif (kon =="W"):
            view = pd.DataFrame(data=data_t[kon].tolist(),columns=["C","i","f","o"]).T
          else:
            view = pd.DataFrame(data=data_t[kon].tolist(),columns=[kon]).T
    elif jenis_proses == "backward":
        print ("PROSES BACKWARD")
        if urutan_ke != "print full":
          print ("tabel hasil perhitungan %s urutan ke %s"%(kon,urutan_ke))
          if kon == "dE_dW_t":
              view = pd.DataFrame(data=data_t.loc[urutan_ke,kon],index=["C","i","f","o"], columns=["bias","date","x(close)","h(close)"])
          elif kon == "dE_dz_t":
              view = pd.DataFrame(data=data_t.loc[urutan_ke,kon],index=["dE_dzcbar_t", "dE_dzi_t", "dE_dzf_t", "dE_dzo_t"],columns=[urutan_ke])
          elif kon == "dE_dI_t":
              view = pd.DataFrame(data=data_t.loc[urutan_ke,kon],index=["bias", "date", "x(close)", "h(Close)"],columns=[urutan_ke])
          else:
              view = pd.DataFrame(data=data_t.loc[urutan_ke,kon],index=[kon],columns=[urutan_ke]).T
        else:
          print ("tabel hasil perhitungan %s urutan ke %s"%(kon,urutan_ke))
          if kon == "dE_dW_t":
            view = pd.DataFrame(data=data_t[kon].tolist(),columns=["C","i","f","o"]).T
          elif kon == "dE_dz_t":
            view = pd.DataFrame(data=data_t[kon].tolist(),columns=["dE_dzcbar_t", "dE_dzi_t", "dE_dzf_t", "dE_dzo_t"]).T
          elif kon == "dE_dI_t":
            view = pd.DataFrame(data=data_t[kon].tolist(),columns=["bias", "date", "x(close)", "h(Close)"]).T
          else:
            view = pd.DataFrame(data=data_t[kon].tolist(),columns=[kon]).T
    elif jenis_proses == "update bobot":
      print ("UPDATE BOBOT")
      if urutan_ke != "print full":
        print ("tabel hasil perhitungan %s urutan ke %s"%(kon,urutan_ke))
        if kon == "new_dE_dW_t":
          view = pd.DataFrame(data=data_t.loc[urutan_ke,kon],index=["C","i","f","o"], columns=["bias","date","x(close)","h(close)"])
      else:
        view = pd.DataFrame(data=data_t[kon].tolist(),columns=["C","i","f","o"]).T
    elif jenis_proses == "optimasi":
      print ("OPTIMASI")
      if urutan_ke != "print full":
        print ("tabel hasil perhitungan %s litrasi ke %s"%(kon,urutan_ke))
        if kon == "optimasi_bobot":
          view = pd.DataFrame(data=data_t.loc[urutan_ke,kon],index=["C","i","f","o"], columns=["bias","date","x(close)","h(close)"])
      else:
        view = pd.DataFrame(data=data_t[kon].tolist(),columns=["C","i","f","o"]).T
    else:
      if urutan_ke != "print full":
        print ("tabel hasil perhitungan %s data ke %s"%(kon,urutan_ke))
        if kon == "data":
          view = pd.DataFrame(data=data_t.loc[urutan_ke,kon], columns=["bias","date","x(close)"])
      else:
        view = pd.DataFrame(data=data_t[kon].tolist()).T
    print ("---------------------------------------")
    return(view)
