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

def get_all_tasks(table, user_name):

    conn  = connect()
    cursor = conn.cursor()
    query = f"SELECT * FROM {table} WHERE user_name = %s;"
    print(query)
    # Execute the query
    cursor.execute(query, (user_name,))
    columns = list(cursor.description)
    result = cursor.fetchall()

    # make dict
    results = []
    for row in result:
        row_dict = {}
        for i, col in enumerate(columns):
            row_dict[col.name] = row[i]
        results.append(row_dict)

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

def get_results(table, id):

    conn = connect()
    cursor = conn.cursor()

    query = f"SELECT * FROM {table} WHERE task_id = %s ORDER BY comm_round DESC;"
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

    query = f"SELECT * FROM task WHERE user_name = %s AND task_name = %s;"
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
