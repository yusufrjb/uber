"""
add_duration_column.py
Script untuk menambahkan kolom 'duration' ke segmented_trips.csv
Duration dihitung dari avg_vtat (Vehicle Time At Trip)
"""

import pandas as pd
import os

# Path files
SEG_PATH = "outputs_customer_segmentation_k_2/segmented_customers.csv"

def add_duration_column():
    """
    Menambahkan kolom duration ke segmented_trips.csv
    Duration = avg_vtat (waktu perjalanan kendaraan dalam menit)
    """
    
    if not os.path.exists(SEG_PATH):
        print(f"❌ File {SEG_PATH} tidak ditemukan!")
        return
    
    print(f"📂 Loading data dari {SEG_PATH}...")
    df = pd.read_csv(SEG_PATH)
    
    print(f"📊 Shape data: {df.shape}")
    print(f"📋 Kolom yang tersedia: {df.columns.tolist()}")
    
    # Cek apakah kolom avg_vtat ada
    if 'avg_vtat' in df.columns:
        print("\n✅ Kolom 'avg_vtat' ditemukan!")
        
        # Tambahkan kolom duration
        # avg_vtat adalah durasi perjalanan dalam menit
        df['duration'] = df['avg_vtat']
        
        # Handle missing values - set ke 0 atau median
        missing_count = df['duration'].isna().sum()
        if missing_count > 0:
            print(f"⚠️  Ditemukan {missing_count} missing values di duration")
            print("   Mengisi dengan 0 untuk data yang hilang...")
            df['duration'] = df['duration'].fillna(0)
        
        print(f"\n📊 Statistik Duration:")
        print(f"   Min: {df['duration'].min():.2f} menit")
        print(f"   Max: {df['duration'].max():.2f} menit")
        print(f"   Mean: {df['duration'].mean():.2f} menit")
        print(f"   Median: {df['duration'].median():.2f} menit")
        
        # Simpan kembali
        df.to_csv(SEG_PATH, index=False)
        print(f"\n💾 File berhasil disimpan dengan kolom 'duration'!")
        print(f"📍 Lokasi: {SEG_PATH}")
        
    elif 'avg_ctat' in df.columns:
        print("\n✅ Kolom 'avg_ctat' ditemukan!")
        print("   Menggunakan avg_ctat sebagai duration...")
        
        df['duration'] = df['avg_ctat']
        
        missing_count = df['duration'].isna().sum()
        if missing_count > 0:
            print(f"⚠️  Ditemukan {missing_count} missing values di duration")
            print("   Mengisi dengan 0 untuk data yang hilang...")
            df['duration'] = df['duration'].fillna(0)
        
        print(f"\n📊 Statistik Duration:")
        print(f"   Min: {df['duration'].min():.2f} menit")
        print(f"   Max: {df['duration'].max():.2f} menit")
        print(f"   Mean: {df['duration'].mean():.2f} menit")
        print(f"   Median: {df['duration'].median():.2f} menit")
        
        df.to_csv(SEG_PATH, index=False)
        print(f"\n💾 File berhasil disimpan dengan kolom 'duration'!")
        print(f"📍 Lokasi: {SEG_PATH}")
        
    else:
        print("\n❌ Tidak ditemukan kolom 'avg_vtat' atau 'avg_ctat'!")
        print("   Kolom yang tersedia:")
        for col in df.columns:
            print(f"   - {col}")
        print("\n💡 Solusi alternatif:")
        print("   1. Gunakan kolom lain yang ada untuk durasi")
        print("   2. Atau hapus requirement duration dari dashboard")

    # Tampilkan sample data
    print("\n📋 Sample data (5 baris pertama):")
    display_cols = ['booking_id', 'avg_vtat', 'duration', 'ride_distance', 'booking_value']
    available_cols = [col for col in display_cols if col in df.columns]
    print(df[available_cols].head())


def add_distance_column():
    """
    Memastikan kolom distance tersedia
    Gunakan ride_distance jika tersedia
    """
    
    if not os.path.exists(SEG_PATH):
        print(f"❌ File {SEG_PATH} tidak ditemukan!")
        return
    
    print(f"\n📂 Checking kolom distance...")
    df = pd.read_csv(SEG_PATH)
    
    if 'distance' not in df.columns:
        if 'ride_distance' in df.columns:
            print("✅ Menambahkan kolom 'distance' dari 'ride_distance'")
            df['distance'] = df['ride_distance']
            
            missing_count = df['distance'].isna().sum()
            if missing_count > 0:
                print(f"⚠️  Ditemukan {missing_count} missing values di distance")
                print("   Mengisi dengan 0 untuk data yang hilang...")
                df['distance'] = df['distance'].fillna(0)
            
            print(f"\n📊 Statistik Distance:")
            print(f"   Min: {df['distance'].min():.2f} km")
            print(f"   Max: {df['distance'].max():.2f} km")
            print(f"   Mean: {df['distance'].mean():.2f} km")
            
            df.to_csv(SEG_PATH, index=False)
            print(f"💾 Kolom 'distance' berhasil ditambahkan!")
        else:
            print("❌ Kolom 'ride_distance' tidak ditemukan!")
    else:
        print("✅ Kolom 'distance' sudah ada")


if __name__ == "__main__":
    print("="*60)
    print("  ADD DURATION & DISTANCE COLUMNS")
    print("="*60)
    
    # Tambahkan kolom duration
    add_duration_column()
    
    print("\n" + "="*60)
    
    # Tambahkan kolom distance
    add_distance_column()
    
    print("\n" + "="*60)
    print("✅ Proses selesai!")
    print("🔄 Silakan refresh dashboard Streamlit untuk melihat perubahan")
    print("="*60)