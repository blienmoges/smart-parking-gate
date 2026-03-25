import sqlite3

DB_PATH = "licensePlatesDatabase.db"

def setup_and_add():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Manually create the table if it's missing
    print("Checking database structure...")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS RegisteredUsers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE NOT NULL,
            owner_name TEXT NOT NULL
        )
    """)
    
    # 2. Add the users
    users_to_add = [
        ('AA12345', 'Tesfaye'),
        ('ETH789', 'Admin_User'),
        ('TEST001', 'Test_Car'),
        ('R94529A' ,"blien"),
        ("A45699", "bb"),
        ("E99999" , "home")
        ,("SN66XMZ" , "nahom") 
    ]

    for plate, name in users_to_add:
        try:
            cursor.execute("INSERT INTO RegisteredUsers (plate_number, owner_name) VALUES (?, ?)", (plate, name))
            print(f"✅ Added: {name} ({plate})")
        except sqlite3.IntegrityError:
            print(f"ℹ️ Already exists: {plate}")

    conn.commit()
    conn.close()
    print("\n🚀 Database is ready for your API!")

if __name__ == "__main__":
    setup_and_add()