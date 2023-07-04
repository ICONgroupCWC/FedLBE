import psycopg2


def connect():
    conn = psycopg2.connect(
        host="localhost",
        port="5432",
        database="fedDB",
        user="postgres",
        password="postgres"
    )

    return conn

def insert(table, values):

    conn  = connect()
    cursor = conn.cursor()
    columns = ", ".join(values.keys())
    placeholders = ", ".join("%s" for _ in values)
    query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
    print(query)
    # Execute the query
    cursor.execute(query, tuple(values.values()))

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()

def get_all(table):

    conn  = connect()
    cursor = conn.cursor()
    query = f"SELECT * FROM {table}"
    print(query)
    # Execute the query
    cursor.execute(query)
    results = cursor.fetchall()
    # Commit the changes to the database
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()
    return results

def get_from_id(table, id):

    conn = connect()
    cursor = conn.cursor()

    query = f"SELECT * FROM {table} WHERE task_id = %s;"
    print(query)
    # Execute the query
    cursor.execute(query, (id,))
    results = cursor.fetchall()
    # Commit the changes to the database
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()
    return results

def get_task_id(user_name, task_name):

    conn = connect()
    cursor = conn.cursor()

    query = f"SELECT * FROM task WHERE user_name = %s AND name = %s;"
    print(query)
    # Execute the query
    cursor.execute(query, (user_name,task_name))
    task_id = cursor.fetchone()[0]

    # Commit the changes to the database
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()
    return task_id
