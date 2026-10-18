package com.paddycare.ai.data

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface ScanDao {
    @Query("SELECT * FROM scan_history ORDER BY timestamp DESC LIMIT 50")
    fun getAllScans(): Flow<List<ScanRecord>>

    @Insert
    suspend fun insert(record: ScanRecord)

    @Query("SELECT imagePath FROM scan_history WHERE id NOT IN (SELECT id FROM scan_history ORDER BY timestamp DESC LIMIT :keepCount)")
    suspend fun getOverflowImagePaths(keepCount: Int = 50): List<String>

    @Query("DELETE FROM scan_history WHERE id NOT IN (SELECT id FROM scan_history ORDER BY timestamp DESC LIMIT :keepCount)")
    suspend fun deleteOverflowRecords(keepCount: Int = 50)

    @Query("DELETE FROM scan_history")
    suspend fun deleteAll()
}
