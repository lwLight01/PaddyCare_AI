package com.paddycare.ai.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface ScanDao {
    @Query("SELECT * FROM scan_history ORDER BY timestamp DESC LIMIT 50")
    fun getAllScans(): Flow<List<ScanRecord>>

    @Insert
    suspend fun insert(record: ScanRecord)

    @Query("DELETE FROM scan_history")
    suspend fun deleteAll()
}
